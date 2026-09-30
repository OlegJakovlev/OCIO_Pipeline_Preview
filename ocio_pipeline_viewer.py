#!/usr/bin/env python3
"""
How rendering works: OpenGL shades in floating point in the renderer's working space
(MSAA float framebuffer), the result is read back and OCIO applies the selected
display/view transform on the CPU. While you drag, it renders at half resolution and
refines to full resolution when you stop.

"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must be set before cv2 import

import PyOpenColorIO as OCIO
from OpenGL import GL as gl
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets


# Candidate colour-space names/aliases across the built-in ACES CG/Studio configs
CS_SRGB_TEXTURE = ["sRGB - Texture", "srgb_tx", "sRGB Encoded Rec.709 (sRGB)", "Utility - sRGB - Texture"]
CS_LIN_709 = ["Linear Rec.709 (sRGB)", "lin_rec709_srgb", "scene-linear Rec.709-sRGB", "Utility - Linear - sRGB"]
CS_ACESCG = ["ACEScg", "ACES - ACEScg", "acescg"]
DISPLAY_CANDIDATES = ["sRGB - Display", "sRGB"]


# --------------------------------------------------------------------------- #
# OCIO helper
# --------------------------------------------------------------------------- #
class OcioContext:
    def __init__(self, config: OCIO.Config) -> None:
        self.config = config
        self.srgb_tex = self._find_cs(CS_SRGB_TEXTURE, "sRGB texture")
        self.lin709 = self._find_cs(CS_LIN_709, "Linear Rec.709")
        self.acescg = self._find_cs(CS_ACESCG, "ACEScg")

        displays = list(config.getDisplays())
        self.display = next((d for d in DISPLAY_CANDIDATES if d in displays), None) \
            or config.getDefaultDisplay()
        self.views = list(config.getViews(self.display))
        self.default_view = config.getDefaultView(self.display)
        self._display_cpu: dict[tuple[str, str], OCIO.CPUProcessor] = {}

    def _find_cs(self, names: list[str], what: str) -> str:
        for n in names:
            cs = self.config.getColorSpace(n)
            if cs is not None:
                return cs.getName()
        raise RuntimeError(f"Config has no colour space for '{what}' (tried {names})")

    @staticmethod
    def _apply(cpu: OCIO.CPUProcessor, arr: np.ndarray) -> np.ndarray:
        out = np.ascontiguousarray(arr, dtype=np.float32).copy()
        h, w, _ = out.shape
        cpu.apply(OCIO.PackedImageDesc(out, w, h, 3))
        return out

    def convert(self, arr: np.ndarray, src: str, dst: str) -> np.ndarray:
        cpu = self.config.getProcessor(src, dst).getDefaultCPUProcessor()
        return self._apply(cpu, arr)

    def to_display(self, arr: np.ndarray, src: str, view: str) -> np.ndarray:
        cpu = self._display_cpu.get((src, view))
        if cpu is None:
            dvt = OCIO.DisplayViewTransform()
            dvt.setSrc(src)
            dvt.setDisplay(self.display)
            dvt.setView(view)
            cpu = self.config.getProcessor(dvt).getDefaultCPUProcessor()
            self._display_cpu[(src, view)] = cpu
        return self._apply(cpu, arr)

    def gamut_matrix(self) -> np.ndarray:
        """3x3 matrix Linear Rec.709 -> ACEScg as OCIO computes it (row-major, out = M @ rgb)."""
        basis = np.eye(3, dtype=np.float32).reshape(1, 3, 3)  # three RGB pixels
        cols = self.convert(basis, self.lin709, self.acescg).reshape(3, 3)  # row i = M @ e_i
        return cols.T.astype(np.float64)


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def box_downscale(arr: np.ndarray, max_size: int) -> np.ndarray:
    h, w, _ = arr.shape
    k = int(np.ceil(max(h, w) / max_size))
    if k <= 1:
        return arr
    h2, w2 = h // k * k, w // k * k
    return arr[:h2, :w2].reshape(h2 // k, k, w2 // k, k, 3).mean(axis=(1, 3)).astype(np.float32)


def read_with_cv2(path: str) -> np.ndarray:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Reading this file needs `pip install OpenEXR` (EXR) "
                           "or `pip install opencv-python`") from e
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"Could not read {path}")
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return np.ascontiguousarray(a[..., :3][..., ::-1], dtype=np.float32)  # BGR -> RGB


def read_exr(path: str) -> np.ndarray:
    """Read an EXR as float32 RGB (half/float, RGB/RGBA/greyscale, separate R,G,B channels).

    Uses the `OpenEXR` package (>= 3.3) when installed, otherwise falls back to OpenCV.
    Alpha is dropped. Multi-part / deep files: only the first part's colour channels are read.
    """
    try:
        import OpenEXR
    except ImportError:
        return read_with_cv2(path)
    if not hasattr(OpenEXR, "File"):  # old OpenEXR (<3.3) without the File API
        return read_with_cv2(path)

    with OpenEXR.File(path) as f:
        ch = f.channels()
        if "RGB" in ch:
            a = ch["RGB"].pixels
        elif "RGBA" in ch:
            a = ch["RGBA"].pixels[..., :3]
        elif all(c in ch for c in "RGB"):
            a = np.stack([ch[c].pixels for c in "RGB"], -1)
        elif "Y" in ch:
            a = np.stack([ch["Y"].pixels] * 3, -1)
        else:  # unusual layer names: take the first three channels alphabetically
            names = sorted(ch)[:3]
            if not names:
                raise RuntimeError("EXR contains no channels")
            names += [names[-1]] * (3 - len(names))
            a = np.stack([ch[n].pixels for n in names], -1)
        return np.ascontiguousarray(a, dtype=np.float32)  # half -> float32


def load_image(path: str, max_size: int = 1024) -> tuple[np.ndarray, bool]:
    """Returns (float32 RGB array, is_float_file). Float files (EXR/HDR) are normally linear."""
    ext = Path(path).suffix.lower()
    if ext == ".exr":
        return box_downscale(read_exr(path), max_size), True
    if ext == ".hdr":
        return box_downscale(read_with_cv2(path), max_size), True

    img = Image.open(path)
    if img.mode.startswith("I"):  # 16-bit greyscale
        a = np.asarray(img, dtype=np.float32) / 65535.0
        a = np.stack([a] * 3, -1)
    else:
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return box_downscale(a, max_size), False


def to_qimage(u8: np.ndarray) -> QtGui.QImage:
    u8 = np.ascontiguousarray(u8)
    h, w, _ = u8.shape
    return QtGui.QImage(u8.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


# --------------------------------------------------------------------------- #
# Settings (all tunable "magic numbers"; edited in the Settings tab, saved with QSettings)
# --------------------------------------------------------------------------- #
@dataclass
class SettingsData:
    # rendering quality
    max_render_size: int = 720
    interactive_scale: float = 0.5
    refine_delay_ms: int = 200
    msaa_samples: int = 4
    anisotropy: float = 8.0
    mesh_detail: int = 64
    max_texture_size: int = 1024
    # stage thumbnails
    thumb_source_size: int = 256
    thumb_width: int = 240
    thumb_height: int = 180
    thumb_render_width: int = 360
    thumb_render_height: int = 270
    # camera
    fov_deg: float = 35.0
    cam_yaw_deg: float = 25.0
    cam_pitch_deg: float = 15.0
    cam_distance: float = 3.8
    orbit_sensitivity: float = 0.008
    zoom_step: float = 0.88
    # shading (colours are Linear Rec.709; converted for each renderer)
    key_color: tuple = (3.0, 2.3, 1.6)
    fill_color: tuple = (0.35, 0.55, 1.0)
    ambient: tuple = (0.10, 0.10, 0.10)
    background: tuple = (0.03, 0.03, 0.035)
    key_dir: tuple = (-0.5, 0.6, 0.65)
    fill_dir: tuple = (0.7, -0.1, 0.4)
    spec_strength: float = 0.08
    spec_power: float = 80.0
    # behaviour
    texture_debounce_ms: int = 40
    paste_column_major: bool = False
    default_config: str = "ocio://cg-config-latest"


# (group title, [(field, label, kind, min, max, step, tooltip)])   kind: int|float|vec3|bool|str|choice:a,b,c
SETTING_GROUPS = [
    ("Rendering quality", [
        ("max_render_size", "Max render size (px, long side)", "int", 64, 4096, 16,
         "Upper limit for the resolution of the 3D viewports."),
        ("interactive_scale", "Resolution scale while dragging", "float", 0.1, 1.0, 0.05,
         "Viewports render at this fraction of full size while you orbit/zoom/pan."),
        ("refine_delay_ms", "Full-quality refine delay (ms)", "int", 0, 5000, 10,
         "Time after the last mouse movement before re-rendering at full resolution."),
        ("msaa_samples", "MSAA samples", "choice:0,2,4,8", 0, 0, 0,
         "Multisample anti-aliasing. Falls back to 0 if the GPU cannot do it."),
        ("anisotropy", "Anisotropic filtering", "float", 1, 16, 1, "Texture sharpness at grazing angles."),
        ("mesh_detail", "Mesh detail (latitude segments)", "int", 8, 256, 8,
         "Segments for sphere / cylinder / torus (longitude uses twice as many)."),
        ("max_texture_size", "Max texture size (px) - applies to next loaded image", "int", 64, 8192, 64,
         "Loaded images are box-downscaled so their long side is at most this."),
    ]),
    ("Stage thumbnails (Pipeline tab)", [
        ("thumb_source_size", "Source image size for thumbnails (px)", "int", 32, 1024, 16,
         "The pipeline is re-run on a copy this size for the stage previews."),
        ("thumb_width", "Thumbnail width (px)", "int", 80, 600, 10, ""),
        ("thumb_height", "Thumbnail height (px)", "int", 60, 600, 10, ""),
        ("thumb_render_width", "Stage render width (px)", "int", 64, 1024, 10,
         "Resolution of stage cards set to 'sRGB / ACEScg renderer'."),
        ("thumb_render_height", "Stage render height (px)", "int", 64, 1024, 10, ""),
    ]),
    ("Camera", [
        ("fov_deg", "Field of view (deg)", "float", 5, 120, 1, ""),
        ("cam_yaw_deg", "Default yaw (deg)", "float", -360, 360, 5, "Used by 'Reset camera' and the stage renders."),
        ("cam_pitch_deg", "Default pitch (deg)", "float", -89, 89, 5, ""),
        ("cam_distance", "Default distance", "float", 0.5, 40, 0.1, ""),
        ("orbit_sensitivity", "Orbit sensitivity (rad/px)", "float", 0.0005, 0.05, 0.001, ""),
        ("zoom_step", "Zoom factor per wheel notch", "float", 0.5, 0.99, 0.01, "Smaller = faster zoom."),
    ]),
    ("Shading and lights (Linear Rec.709, converted for each renderer)", [
        ("key_color", "Key light colour", "vec3", 0, 100, 0.1, ""),
        ("fill_color", "Fill light colour", "vec3", 0, 100, 0.05, ""),
        ("ambient", "Ambient", "vec3", 0, 100, 0.01, ""),
        ("background", "Background", "vec3", 0, 100, 0.005, ""),
        ("key_dir", "Key light direction (towards light)", "vec3", -10, 10, 0.1, "Normalised automatically."),
        ("fill_dir", "Fill light direction (towards light)", "vec3", -10, 10, 0.1, ""),
        ("spec_strength", "Specular strength", "float", 0, 5, 0.01, ""),
        ("spec_power", "Specular power", "float", 1, 1000, 5, ""),
    ]),
    ("Behaviour", [
        ("texture_debounce_ms", "Texture rebuild delay (ms)", "int", 0, 2000, 10,
         "Delay after a control changes before the OCIO texture is rebuilt."),
        ("paste_column_major", "Pasted matrices are column-major", "bool", 0, 0, 0,
         "If on, a pasted matrix is transposed after parsing."),
        ("default_config", "Default OCIO config - applies on next start", "str", 0, 0, 0,
         "Used when $OCIO is not set. Example: ocio://cg-config-latest or a path to a .ocio file."),
    ]),
]


class Settings(QtCore.QObject):
    """Holds SettingsData (`.d`, a stable object edited in place) and persists it as JSON."""
    changed = QtCore.Signal(str)  # field name, or "*" for a full reset

    def __init__(self) -> None:
        super().__init__()
        self.d = SettingsData()
        self._store = QtCore.QSettings("OCIOPipelineViewer", "OCIOPipelineViewer")
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(str(self._store.value("settings_json", "") or "{}"))
        except (ValueError, TypeError):
            return
        for f in fields(SettingsData):
            if f.name not in raw:
                continue
            default, v = getattr(self.d, f.name), raw[f.name]
            try:
                if isinstance(default, tuple):
                    v = tuple(float(x) for x in v)
                    if len(v) != len(default):
                        continue
                elif isinstance(default, bool):
                    v = bool(v)
                else:
                    v = type(default)(v)
            except (ValueError, TypeError):
                continue
            setattr(self.d, f.name, v)

    def save(self) -> None:
        self._store.setValue("settings_json", json.dumps(asdict(self.d)))

    def set(self, name: str, value) -> None:
        setattr(self.d, name, value)
        self.save()
        self.changed.emit(name)

    def reset(self) -> None:
        for f in fields(SettingsData):
            setattr(self.d, f.name, getattr(SettingsData(), f.name))
        self.save()
        self.changed.emit("*")


def scene_from_settings(d: SettingsData) -> np.ndarray:
    """Rows: key, fill, ambient, background (Linear Rec.709)."""
    return np.array([d.key_color, d.fill_color, d.ambient, d.background], dtype=np.float32)


def unit_vector(v: tuple) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-8 else np.array([0, 0, 1], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Matrix parsing (clipboard text -> 3x3)
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_BRACKET_GROUP = re.compile(r"[\[\(\{]([^\[\]\(\)\{\}]*)[\]\)\}]")


def parse_matrix(text: str, column_major: bool = False) -> np.ndarray:
    """Parse a 3x3 matrix from text. Supported layouts (numbers may use exponents):

    * spaced/tabbed columns, one row per line        ``0.6 0.3 0.1`` / ``0.1 0.8 0.1`` / ...
    * brackets and commas (Python/NumPy/JSON/GLSL-like) ``[[0.6, 0.3, 0.1], [...], [...]]``
    * ``;`` as row separator (MATLAB style), or a flat list of 9 numbers
    * 4x4 (16 numbers) or 3x4 (12 numbers): the top-left 3x3 is used

    Values are row-major unless `column_major` is set (then the result is transposed).
    """
    t = text.replace("\u2212", "-").strip()
    if not t:
        raise ValueError("The clipboard is empty.")
    groups = _BRACKET_GROUP.findall(t)
    if groups:
        chunks = groups
    else:
        chunks = [ln for ln in re.split(r"[\n\r;]+", t)]
    rows = [nums for nums in ([float(x) for x in _NUM.findall(c)] for c in chunks) if nums]

    if len(rows) >= 3 and all(len(r) >= 3 for r in rows[:3]):
        m = [r[:3] for r in rows[:3]]
    else:
        flat = [x for r in rows for x in r]
        if len(flat) == 9:
            m = np.array(flat).reshape(3, 3)
        elif len(flat) == 16:
            m = np.array(flat).reshape(4, 4)[:3, :3]
        elif len(flat) == 12:
            m = np.array(flat).reshape(3, 4)[:, :3]
        else:
            raise ValueError(f"Could not find a 3x3 matrix in the clipboard (found {len(flat)} numbers "
                             "in an unrecognised layout).")
    arr = np.array(m, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError("The parsed matrix is not a finite 3x3 matrix.")
    return (arr.T if column_major else arr).astype(np.float32)


# --------------------------------------------------------------------------- #
# Procedural meshes: vertices are (pos xyz, normal xyz, uv) float32, indices uint32.
# UV origin is the top-left of the image (v grows downwards), matching row 0 of the texture.
# --------------------------------------------------------------------------- #
PRIMITIVES = ["Sphere", "Plane", "Cube", "Cylinder", "Torus"]
TAU = 2 * np.pi


def _pack(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, nrm, uv], -1).reshape(-1, 8).astype(np.float32)


def _grid_indices(rows: int, cols: int) -> np.ndarray:
    i, j = np.mgrid[0:rows, 0:cols]
    a = i * (cols + 1) + j
    b = a + cols + 1
    return np.stack([a, b, a + 1, a + 1, b, b + 1], -1).reshape(-1).astype(np.uint32)


def mesh_sphere(_aspect: float, detail: int = 64) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = detail, 2 * detail
    v = np.linspace(0, 1, rows + 1)[:, None]
    u = np.linspace(0, 1, cols + 1)[None, :]
    th, ph = v * np.pi, (u - 0.5) * TAU  # u = 0.5 faces +Z
    x = np.sin(th) * np.sin(ph)
    y = np.cos(th) + 0 * ph
    z = np.sin(th) * np.cos(ph)
    pos = np.stack(np.broadcast_arrays(x, y, z), -1)
    uv = np.stack(np.broadcast_arrays(u, v), -1)
    return _pack(pos, pos, uv), _grid_indices(rows, cols)


def mesh_torus(_aspect: float, detail: int = 64, big_r: float = 0.72,
               small_r: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = detail, 2 * detail
    v = np.linspace(0, 1, rows + 1)[:, None]
    u = np.linspace(0, 1, cols + 1)[None, :]
    ph, th = (u - 0.5) * TAU, (0.5 - v) * TAU
    rr = big_r + small_r * np.cos(th)
    x, y, z = rr * np.sin(ph), small_r * np.sin(th) + 0 * ph, rr * np.cos(ph)
    nx, ny, nz = np.cos(th) * np.sin(ph), np.sin(th) + 0 * ph, np.cos(th) * np.cos(ph)
    pos = np.stack(np.broadcast_arrays(x, y, z), -1)
    nrm = np.stack(np.broadcast_arrays(nx, ny, nz), -1)
    uv = np.stack(np.broadcast_arrays(u, v), -1)
    return _pack(pos, nrm, uv), _grid_indices(rows, cols)


def mesh_plane(aspect: float, _detail: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ax, ay = min(1.0, aspect), min(1.0, 1.0 / aspect)
    pos = np.array([[-ax, ay, 0], [ax, ay, 0], [-ax, -ay, 0], [ax, -ay, 0]], dtype=np.float64)
    nrm = np.tile([0.0, 0.0, 1.0], (4, 1))
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    return _pack(pos, nrm, uv), np.array([0, 2, 1, 1, 2, 3], dtype=np.uint32)


def mesh_cube(_aspect: float, _detail: int = 0, h: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    faces = [  # (normal, up)
        ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, -1)), ((0, -1, 0), (0, 0, 1)),
    ]
    verts, idx = [], []
    for k, (n, up) in enumerate(faces):
        n, up = np.array(n, float), np.array(up, float)
        right = np.cross(up, n)
        for sr, su, uu, vv in ((-1, 1, 0, 0), (1, 1, 1, 0), (-1, -1, 0, 1), (1, -1, 1, 1)):
            verts.append(np.concatenate([h * (n + sr * right + su * up), n, [uu, vv]]))
        b = 4 * k
        idx += [b, b + 2, b + 1, b + 1, b + 2, b + 3]
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


def mesh_cylinder(_aspect: float, detail: int = 64, r: float = 0.6,
                  h: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    cols = 2 * detail
    u = np.linspace(0, 1, cols + 1)
    ph = (u - 0.5) * TAU
    sn, cs = np.sin(ph), np.cos(ph)
    zero = np.zeros_like(u)
    top = np.stack([r * sn, zero + h, r * cs], -1)
    bot = np.stack([r * sn, zero - h, r * cs], -1)
    nrm = np.stack([sn, zero, cs], -1)
    pos = np.stack([top, bot], 0)                       # (2, cols+1, 3)
    nrms = np.stack([nrm, nrm], 0)
    uv = np.stack([np.stack([u, zero], -1), np.stack([u, zero + 1], -1)], 0)
    verts = [_pack(pos, nrms, uv)]
    idx = [_grid_indices(1, cols)]
    offset = 2 * (cols + 1)
    for s in (1.0, -1.0):                               # caps as triangle fans
        centre = np.array([[0, s * h, 0, 0, s, 0, 0.5, 0.5]], dtype=np.float32)
        ring = _pack(np.stack([r * sn, zero + s * h, r * cs], -1), np.tile([0, s, 0], (cols + 1, 1)),
                     np.stack([0.5 + sn / 2, 0.5 + s * cs / 2], -1))
        verts += [centre, ring]
        k = np.arange(cols)
        idx.append(np.stack([np.full(cols, offset), offset + 1 + k, offset + 2 + k], -1)
                   .reshape(-1).astype(np.uint32))
        offset += 1 + cols + 1
    return np.concatenate(verts), np.concatenate(idx)


MESH_BUILDERS = {"Sphere": mesh_sphere, "Plane": mesh_plane, "Cube": mesh_cube,
                 "Cylinder": mesh_cylinder, "Torus": mesh_torus}


# --------------------------------------------------------------------------- #
# Camera (shared by both viewports)
# --------------------------------------------------------------------------- #
class Camera(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self, d: SettingsData) -> None:
        super().__init__()
        self.d = d
        self.reset(emit=False)

    @property
    def fov(self) -> float:
        return float(np.radians(self.d.fov_deg))

    def reset(self, emit: bool = True) -> None:
        self.yaw, self.pitch = np.radians(self.d.cam_yaw_deg), np.radians(self.d.cam_pitch_deg)
        self.dist = float(self.d.cam_distance)
        self.target = np.zeros(3)
        if emit:
            self.changed.emit()

    def eye(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return self.target + self.dist * np.array([cp * np.sin(self.yaw), np.sin(self.pitch),
                                                   cp * np.cos(self.yaw)])

    def view(self) -> np.ndarray:
        e = self.eye()
        f = self.target - e
        f /= np.linalg.norm(f)
        s = np.cross(f, [0, 1, 0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -m[:3, :3] @ e
        return m

    def proj(self, aspect: float, near: float = 0.05, far: float = 100.0) -> np.ndarray:
        f = 1.0 / np.tan(self.fov / 2)
        m = np.zeros((4, 4))
        m[0, 0], m[1, 1] = f / aspect, f
        m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
        m[3, 2] = -1.0
        return m

    def orbit(self, dx: float, dy: float) -> None:
        k = self.d.orbit_sensitivity
        self.yaw -= dx * k
        self.pitch = float(np.clip(self.pitch + dy * k, np.radians(-89), np.radians(89)))
        self.changed.emit()

    def zoom(self, notches: float) -> None:
        self.dist = float(np.clip(self.dist * self.d.zoom_step ** notches, 0.4, 40.0))
        self.changed.emit()

    def pan(self, dx: float, dy: float, height_px: float) -> None:
        v = self.view()
        scale = 2 * self.dist * np.tan(self.fov / 2) / max(height_px, 1)
        self.target = self.target + (-dx * v[0, :3] + dy * v[1, :3]) * scale
        self.changed.emit()


# --------------------------------------------------------------------------- #
# OpenGL renderer: off-screen context, float MSAA framebuffer, lit in working space
# --------------------------------------------------------------------------- #
VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec2 aUV;
uniform mat4 uMVP;
out vec3 vPos;
out vec3 vNrm;
out vec2 vUV;
void main() {
    vPos = aPos; vNrm = aNrm; vUV = aUV;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec3 vPos;
in vec3 vNrm;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3 uCam;
uniform vec3 uKey, uFill, uAmb;      // light colours, in the renderer's working space
uniform vec3 uLKey, uLFill;          // directions towards the lights (world space)
uniform vec2 uSpec;                  // specular strength, power
out vec4 outColor;
void main() {
    vec3 N = normalize(vNrm);
    vec3 V = normalize(uCam - vPos);
    if (dot(N, V) < 0.0) N = -N;                       // two-sided (plane)
    vec3 albedo = texture(uTex, vUV).rgb;              // texture numbers used as-is
    vec3 light = uKey * max(dot(N, uLKey), 0.0) + uFill * max(dot(N, uLFill), 0.0) + uAmb;
    vec3 H = normalize(uLKey + V);
    vec3 col = albedo * light + uSpec.x * pow(max(dot(N, H), 0.0), uSpec.y) * uKey;
    outColor = vec4(col, 1.0);
}
"""

def _compile(kind: int, src: str) -> int:
    sh = gl.glCreateShader(kind)
    gl.glShaderSource(sh, src)
    gl.glCompileShader(sh)
    if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
        raise RuntimeError("Shader compile error:\n" + gl.glGetShaderInfoLog(sh).decode(errors="replace"))
    return sh


class GLRenderer:
    def __init__(self, d: SettingsData) -> None:
        self.d = d
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.ctx = QtGui.QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            raise RuntimeError("Could not create an OpenGL 3.3 core context")
        self.surface = QtGui.QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        self.make_current()

        self.info = (gl.glGetString(gl.GL_RENDERER) or b"?").decode(errors="replace")
        vs, fs = _compile(gl.GL_VERTEX_SHADER, VERT_SRC), _compile(gl.GL_FRAGMENT_SHADER, FRAG_SRC)
        self.prog = gl.glCreateProgram()
        gl.glAttachShader(self.prog, vs)
        gl.glAttachShader(self.prog, fs)
        gl.glLinkProgram(self.prog)
        if not gl.glGetProgramiv(self.prog, gl.GL_LINK_STATUS):
            raise RuntimeError("Program link error:\n" + gl.glGetProgramInfoLog(self.prog).decode(errors="replace"))

        self.texture = int(gl.glGenTextures(1))
        self.stage_tex: dict[str, int] = {}
        self.tex_aspect = 1.0
        self.meshes: dict[tuple[str, float], tuple[int, int]] = {}  # key -> (vao, index count)
        self.fbos: dict[tuple[int, int, int], dict[str, int]] = {}

    def make_current(self) -> None:
        self.ctx.makeCurrent(self.surface)

    # ---- textures ----
    def _upload(self, tex_id: int, tex: np.ndarray) -> None:
        h, w, _ = tex.shape
        tex = np.ascontiguousarray(tex, dtype=np.float32)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, w, h, 0, gl.GL_RGB, gl.GL_FLOAT, tex)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        try:  # anisotropic filtering, if available
            gl.glTexParameterf(gl.GL_TEXTURE_2D, 0x84FE, float(self.d.anisotropy))
        except Exception:  # noqa: BLE001
            pass

    def set_texture(self, tex: np.ndarray) -> None:
        """Main texture used by the interactive viewports."""
        self.make_current()
        self.tex_aspect = tex.shape[1] / tex.shape[0]
        self._upload(self.texture, tex)
        self.meshes_to_free()  # the plane mesh depends on the texture aspect

    def set_stage_texture(self, key: str, tex: np.ndarray) -> int:
        """Extra textures (intermediate pipeline stages); returns the GL texture id."""
        self.make_current()
        tid = self.stage_tex.setdefault(key, int(gl.glGenTextures(1)))
        self._upload(tid, tex)
        return tid

    def settings_changed(self, name: str) -> None:
        self.make_current()
        if name in ("msaa_samples", "*"):
            for f in self.fbos.values():
                self._free_fbo(f)
            self.fbos.clear()
        if name in ("mesh_detail", "*"):
            for vao, _n in self.meshes.values():
                gl.glDeleteVertexArrays(1, [vao])
            self.meshes.clear()

    def meshes_to_free(self) -> None:
        for key in [k for k in self.meshes if k[0] == "Plane"]:
            gl.glDeleteVertexArrays(1, [self.meshes.pop(key)[0]])

    # ---- meshes ----
    def _mesh(self, shape: str, aspect: float) -> tuple[int, int]:
        aspect = round(aspect, 3) if shape == "Plane" else 1.0
        key = (shape, aspect, self.d.mesh_detail)
        if key not in self.meshes:
            verts, idx = MESH_BUILDERS[shape](aspect, self.d.mesh_detail)
            vao = int(gl.glGenVertexArrays(1))
            gl.glBindVertexArray(vao)
            vbo, ebo = gl.glGenBuffers(2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
            stride = 8 * 4
            for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(off))
            gl.glBindVertexArray(0)
            self.meshes[key] = (vao, len(idx))
        return self.meshes[key]

    # ---- framebuffers (a few sizes are cached: viewports and stage renders differ) ----
    @staticmethod
    def _free_fbo(f: dict[str, int]) -> None:
        for k in ("ms", "resolve"):
            if k in f:
                gl.glDeleteFramebuffers(1, [f[k]])
        for k in ("rb_color", "rb_depth"):
            if k in f:
                gl.glDeleteRenderbuffers(1, [f[k]])
        if "tex" in f:
            gl.glDeleteTextures(1, [f["tex"]])
        f.clear()

    @staticmethod
    def _create_fbo(f: dict[str, int], w: int, h: int, samples: int) -> None:
        f["ms"] = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f["ms"])
        for key, fmt, att in (("rb_color", gl.GL_RGBA32F, gl.GL_COLOR_ATTACHMENT0),
                              ("rb_depth", gl.GL_DEPTH_COMPONENT24, gl.GL_DEPTH_ATTACHMENT)):
            f[key] = int(gl.glGenRenderbuffers(1))
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, f[key])
            if samples:
                gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, fmt, w, h)
            else:
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, fmt, w, h)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, att, gl.GL_RENDERBUFFER, f[key])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("multisample framebuffer incomplete")

        f["tex"] = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, f["tex"])
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        f["resolve"] = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f["resolve"])
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, f["tex"], 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("resolve framebuffer incomplete")

    def _ensure_fbo(self, w: int, h: int) -> dict[str, int]:
        key = (w, h, self.d.msaa_samples)
        if key in self.fbos:
            return self.fbos[key]
        while len(self.fbos) >= 8:  # drop the oldest
            self._free_fbo(self.fbos.pop(next(iter(self.fbos))))
        last_err: Exception | None = None
        for samples in dict.fromkeys((self.d.msaa_samples, 0)):
            f: dict[str, int] = {}
            try:
                self._create_fbo(f, w, h, samples)
                self.fbos[key] = f
                return f
            except Exception as e:  # noqa: BLE001  (GLError or RuntimeError)
                last_err = e
                self._free_fbo(f)
        raise RuntimeError(f"Could not create a floating-point framebuffer: {last_err}")

    # ---- draw ----
    def render(self, shape: str, cam: Camera, scene: np.ndarray, w: int, h: int,
               texture: int | None = None, aspect: float | None = None) -> np.ndarray:
        """Returns float32 RGB (top-down) in the working space that `scene` is expressed in."""
        self.make_current()
        fb = self._ensure_fbo(w, h)
        vao, count = self._mesh(shape, aspect or self.tex_aspect)
        key, fill, amb, bg = (np.asarray(c, dtype=np.float32) for c in scene)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fb["ms"])
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(float(bg[0]), float(bg[1]), float(bg[2]), 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(self.prog)
        mvp = (cam.proj(w / h) @ cam.view()).astype(np.float32)
        loc = lambda n: gl.glGetUniformLocation(self.prog, n)  # noqa: E731
        gl.glUniformMatrix4fv(loc("uMVP"), 1, gl.GL_TRUE, mvp)
        eye = cam.eye().astype(np.float32)
        gl.glUniform3f(loc("uCam"), *map(float, eye))
        gl.glUniform3f(loc("uKey"), *map(float, key))
        gl.glUniform3f(loc("uFill"), *map(float, fill))
        gl.glUniform3f(loc("uAmb"), *map(float, amb))
        gl.glUniform3f(loc("uLKey"), *map(float, unit_vector(self.d.key_dir)))
        gl.glUniform3f(loc("uLFill"), *map(float, unit_vector(self.d.fill_dir)))
        gl.glUniform2f(loc("uSpec"), float(self.d.spec_strength), float(self.d.spec_power))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture or self.texture)
        gl.glUniform1i(loc("uTex"), 0)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fb["ms"])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fb["resolve"])
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fb["resolve"])
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_FLOAT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)
        return arr.reshape(h, w, 3)[::-1]  # GL is bottom-up


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
class ViewportWidget(QtWidgets.QWidget):
    """Shows the latest rendered frame; mouse input drives the shared camera."""

    def __init__(self, camera: Camera) -> None:
        super().__init__()
        self.camera = camera
        self.image: QtGui.QImage | None = None
        self._last = QtCore.QPointF()
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def pixel_size(self) -> tuple[int, int]:
        dpr = self.devicePixelRatioF()
        return max(16, int(self.width() * dpr)), max(16, int(self.height() * dpr))

    def set_image(self, img: QtGui.QImage) -> None:
        self.image = img
        self.update()

    def paintEvent(self, _e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#111"))
        if self.image is not None:
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(self.rect(), self.image)
        else:
            p.setPen(QtGui.QColor("#666"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Load an image to begin")
        p.setPen(QtGui.QColor(255, 255, 255, 110))
        p.drawText(8, self.height() - 8, "drag: orbit · wheel: zoom · right-drag: pan · double-click: reset")

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self._last = e.position()
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, _e: QtGui.QMouseEvent) -> None:
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        d = e.position() - self._last
        self._last = e.position()
        if e.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.camera.orbit(d.x(), d.y())
        elif e.buttons() & (QtCore.Qt.MouseButton.RightButton | QtCore.Qt.MouseButton.MiddleButton):
            self.camera.pan(d.x(), d.y(), self.height())

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:
        self.camera.zoom(e.angleDelta().y() / 120.0)

    def mouseDoubleClickEvent(self, _e: QtGui.QMouseEvent) -> None:
        self.camera.reset()


class DropSlot(QtWidgets.QLabel):
    fileDropped = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__("Drop image here\nor click to browse")
        self.setAcceptDrops(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QLabel{border:2px dashed #888; border-radius:8px; color:#888;}")
        self._pix: QtGui.QPixmap | None = None

    def set_preview(self, pix: QtGui.QPixmap) -> None:
        self._pix = pix
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size() - QtCore.QSize(8, 8),
                                            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                            QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._rescale()

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.exr *.hdr)")
        if path:
            self.fileDropped.emit(path)


# --------------------------------------------------------------------------- #
# Pipeline stages (used for the final texture and for the Pipeline-tab thumbnails)
# --------------------------------------------------------------------------- #
@dataclass
class Stages:
    inp: np.ndarray      # 1. image as loaded
    inp_cs: str          #    colour space it is interpreted as
    ws: np.ndarray       # 2. after working-space conversion (OCIO decode + OCIO gamut)
    ws_cs: str
    mx: np.ndarray       # 3. after the (optional) matrix override; this is the final texture
    mx_cs: str


def run_pipeline(ctx: OcioContext, image: np.ndarray, is_srgb: bool, convert: bool,
                 override: bool, matrix: np.ndarray) -> Stages:
    inp_cs = ctx.srgb_tex if is_srgb else ctx.lin709
    if not convert:  # values are passed on untouched; read as Linear Rec.709 for display purposes
        return Stages(image, inp_cs, image, ctx.lin709, image, ctx.lin709)
    lin = ctx.convert(image, ctx.srgb_tex, ctx.lin709) if is_srgb else image
    ws = ctx.convert(lin, ctx.lin709, ctx.acescg)
    mx = (lin @ matrix.T).astype(np.float32) if override else ws
    return Stages(image, inp_cs, ws, ctx.acescg, mx, ctx.acescg)


def fmt_rgb(a: np.ndarray) -> str:
    m = np.nan_to_num(a).reshape(-1, 3).mean(0)
    return "mean RGB [" + ", ".join(f"{float(v):.4f}" for v in m) + "]"


class MainWindow(QtWidgets.QMainWindow):
    SHOW_MODES = ["Flat texture", "sRGB renderer", "ACEScg renderer"]
    STAGE_TITLES = {"in": "Input image", "ws": "Working space", "mx": "Input matrix", "view": "View transform"}

    def __init__(self, ctx: OcioContext, renderer: GLRenderer, settings: Settings) -> None:
        super().__init__()
        self.setWindowTitle("OCIO Pipeline Viewer")
        self.ctx = ctx
        self.gl = renderer
        self.settings = settings
        self.d = settings.d
        self.camera = Camera(self.d)
        self.image: np.ndarray | None = None
        self.thumb: np.ndarray | None = None
        self.image_path = ""
        self.tex: np.ndarray | None = None
        self.pipeline_log: list[str] = []
        self.ocio_matrix = np.eye(3)
        self._fast = False
        self.thumb_cam = Camera(self.d)  # fixed camera for the stage renders (not linked to the viewports)
        self.show_combos: dict[str, QtWidgets.QComboBox] = {}
        self.override_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.override_key: str | None = None      # pipeline step shown in the renderers (None = final)
        self.stages_full: Stages | None = None
        self.thumb_labels: list[QtWidgets.QLabel] = []
        self.setting_setters: dict[str, callable] = {}
        self.scene_709 = scene_from_settings(self.d)
        self.scene_acescg = self.scene_709
        self.caps_srgb: list[QtWidgets.QLabel] = []
        self.caps_acescg: list[QtWidgets.QLabel] = []

        self.tex_timer = QtCore.QTimer(singleShot=True, interval=self.d.texture_debounce_ms)
        self.tex_timer.timeout.connect(self.rebuild_texture)
        self.draw_timer = QtCore.QTimer(singleShot=True, interval=8)
        self.draw_timer.timeout.connect(lambda: self.redraw(self._fast))
        self.refine_timer = QtCore.QTimer(singleShot=True, interval=self.d.refine_delay_ms)
        self.refine_timer.timeout.connect(lambda: self.redraw(False))
        self.prev_timer = QtCore.QTimer(singleShot=True, interval=30)
        self.prev_timer.timeout.connect(self.update_previews)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_viewer_tab(), "Viewer")
        self.tabs.addTab(self._build_pipeline_tab(), "Pipeline")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self._wire()
        self.bind_context(ctx)
        self.resize(1400, 880)

    # ------------------------------------------------------------------ UI --
    def _viewport_pair(self) -> tuple[QtWidgets.QHBoxLayout, ViewportWidget, ViewportWidget]:
        row = QtWidgets.QHBoxLayout()
        vps = []
        for caps in (self.caps_srgb, self.caps_acescg):
            col = QtWidgets.QVBoxLayout()
            cap = QtWidgets.QLabel()
            cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cap.setStyleSheet("font-weight:bold;")
            caps.append(cap)
            vp = ViewportWidget(self.camera)
            col.addWidget(cap)
            col.addWidget(vp, 1)
            row.addLayout(col, 1)
            vps.append(vp)
        return row, vps[0], vps[1]

    def _build_viewer_tab(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(root)

        left = QtWidgets.QVBoxLayout()
        lay.addLayout(left, 0)

        g1 = QtWidgets.QGroupBox("1 · Input image")
        l1 = QtWidgets.QVBoxLayout(g1)
        self.slot = DropSlot()
        self.slot.fileDropped.connect(self.load_path)
        self.chk_srgb = QtWidgets.QCheckBox("Treat image as sRGB  (unchecked = linear / raw)")
        self.chk_srgb.setChecked(True)
        l1.addWidget(self.slot)
        l1.addWidget(self.chk_srgb)
        left.addWidget(g1)

        g2 = QtWidgets.QGroupBox("2 · Working-space conversion")
        l2 = QtWidgets.QVBoxLayout(g2)
        self.chk_convert = QtWidgets.QCheckBox("Convert image (sRGB / Raw) → ACEScg")
        self.chk_convert.setChecked(True)
        l2.addWidget(self.chk_convert)
        note = QtWidgets.QLabel("Off = pixel values go to the renderers untouched.\n"
                                "Raw is assumed to have Rec.709 primaries.")
        note.setStyleSheet("color:#888;")
        l2.addWidget(note)
        left.addWidget(g2)

        g3 = QtWidgets.QGroupBox("3 · View transform")
        l3 = QtWidgets.QFormLayout(g3)
        self.cmb_view = QtWidgets.QComboBox()
        self.lbl_display = QtWidgets.QLabel()
        l3.addRow("Display:", self.lbl_display)
        l3.addRow("View:", self.cmb_view)
        left.addWidget(g3)

        g4 = QtWidgets.QGroupBox("4 · Input matrix override (Rec.709 → ACEScg gamut step)")
        l4 = QtWidgets.QVBoxLayout(g4)
        self.chk_matrix = QtWidgets.QCheckBox("Override OCIO gamut matrix")
        l4.addWidget(self.chk_matrix)
        grid = QtWidgets.QGridLayout()
        self.spins: list[QtWidgets.QDoubleSpinBox] = []
        for i in range(9):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(6)
            sb.setRange(-100, 100)
            sb.setSingleStep(0.01)
            sb.valueChanged.connect(self.schedule_texture)
            grid.addWidget(sb, i // 3, i % 3)
            self.spins.append(sb)
        l4.addLayout(grid)
        btn_reset = QtWidgets.QPushButton("Reset to OCIO matrix")
        btn_reset.clicked.connect(self.reset_matrix)
        l4.addWidget(btn_reset)
        btn_paste = QtWidgets.QPushButton("Insert matrix from clipboard")
        btn_paste.setToolTip("Parses spaced columns (one row per line) or [[a, b, c], ...] with brackets and commas")
        btn_paste.clicked.connect(self.paste_matrix)
        l4.addWidget(btn_paste)
        left.addWidget(g4)

        btn_cfg = QtWidgets.QPushButton("Load OCIO config…")
        btn_cfg.clicked.connect(self.load_config_dialog)
        left.addWidget(btn_cfg)
        self.lbl_cfg = QtWidgets.QLabel()
        self.lbl_cfg.setStyleSheet("color:#888;")
        left.addWidget(self.lbl_cfg)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        lay.addLayout(right, 1)
        shape_row = QtWidgets.QHBoxLayout()
        shape_row.addWidget(QtWidgets.QLabel("Primitive:"))
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(PRIMITIVES)
        shape_row.addWidget(self.cmb_shape)
        btn_cam = QtWidgets.QPushButton("Reset camera")
        btn_cam.clicked.connect(self.camera.reset)
        shape_row.addWidget(btn_cam)
        shape_row.addStretch(1)
        right.addLayout(shape_row)

        row, self.view_srgb, self.view_acescg = self._viewport_pair()
        right.addLayout(row, 1)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMaximumHeight(150)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        right.addWidget(self.log)
        return root

    def _card(self, title: str, key: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout, QtWidgets.QLabel]:
        box = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(box)
        thumb = QtWidgets.QLabel()
        thumb.setFixedSize(self.d.thumb_width, self.d.thumb_height)
        self.thumb_labels.append(thumb)
        thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("background:#111; color:#666;")
        thumb.setText("no image")
        v.addWidget(thumb, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        combo = QtWidgets.QComboBox()
        combo.addItems(self.SHOW_MODES)
        combo.setToolTip("Flat texture, or render this stage's texture on the selected primitive "
                         "with the sRGB / ACEScg renderer (through the view transform)")
        form = QtWidgets.QFormLayout()
        form.addRow("Show as:", combo)
        v.addLayout(form)
        self.show_combos[key] = combo
        chk = QtWidgets.QCheckBox("Show this step in the renderers")
        chk.setToolTip("Override: the two 3D renderers use this step's texture instead of the final one")
        chk.toggled.connect(lambda checked, k=key: self.on_override(k, checked))
        v.addWidget(chk)
        self.override_checks[key] = chk
        return box, v, thumb

    @staticmethod
    def _info_label(mono: bool = False) -> QtWidgets.QLabel:
        lb = QtWidgets.QLabel()
        lb.setWordWrap(True)
        lb.setStyleSheet("color:#999; font-size:11px;")
        if mono:
            lb.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        return lb

    @staticmethod
    def _arrow() -> QtWidgets.QLabel:
        a = QtWidgets.QLabel("▶")
        a.setStyleSheet("font-size:22px; color:#888;")
        a.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return a

    def _build_pipeline_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Flat-texture thumbnails show:"))
        self.cmb_preview = QtWidgets.QComboBox()
        self.cmb_preview.addItems(["Raw values (numbers as-is, clamped)",
                                   "Colour-managed (through the view transform)"])
        top.addWidget(self.cmb_preview)
        top.addStretch(1)
        outer.addLayout(top)

        chain = QtWidgets.QHBoxLayout()
        outer.addLayout(chain)

        # 1 · Input image
        c1, v1, self.th_in = self._card("1 · Input image", "in")
        self.cmb_interp = QtWidgets.QComboBox()
        self.cmb_interp.addItems(["sRGB (encoded)", "Linear / Raw"])
        f1 = QtWidgets.QFormLayout()
        f1.addRow("Interpret as:", self.cmb_interp)
        v1.addLayout(f1)
        self.info_in = self._info_label()
        v1.addWidget(self.info_in)
        v1.addStretch(1)

        # 2 · Working space
        c2, v2, self.th_ws = self._card("2 · Working space", "ws")
        self.cmb_conv = QtWidgets.QComboBox()
        self.cmb_conv.addItems(["None (values as-is)", "ACEScg"])
        f2 = QtWidgets.QFormLayout()
        f2.addRow("Convert to:", self.cmb_conv)
        v2.addLayout(f2)
        self.info_ws = self._info_label()
        v2.addWidget(self.info_ws)
        v2.addStretch(1)

        # 3 · Input matrix
        c3, v3, self.th_mx = self._card("3 · Input matrix", "mx")
        self.cmb_mat = QtWidgets.QComboBox()
        self.cmb_mat.addItems(["OCIO default (Rec.709 → ACEScg)", "Custom override"])
        f3 = QtWidgets.QFormLayout()
        f3.addRow("Gamut matrix:", self.cmb_mat)
        v3.addLayout(f3)
        self.info_mx = self._info_label(mono=True)
        v3.addWidget(self.info_mx)
        btn_paste2 = QtWidgets.QPushButton("Insert matrix from clipboard")
        btn_paste2.setToolTip("Parses spaced columns (one row per line) or [[a, b, c], ...] with brackets and commas")
        btn_paste2.clicked.connect(self.paste_matrix)
        v3.addWidget(btn_paste2)
        btn_edit = QtWidgets.QPushButton("Edit matrix values…")
        btn_edit.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        v3.addWidget(btn_edit)
        v3.addStretch(1)

        # 4 · View transform
        c4, v4, self.th_view = self._card("4 · View transform", "view")
        self.cmb_view2 = QtWidgets.QComboBox()
        self.lbl_display2 = QtWidgets.QLabel()
        f4 = QtWidgets.QFormLayout()
        f4.addRow("Display:", self.lbl_display2)
        f4.addRow("View:", self.cmb_view2)
        v4.addLayout(f4)
        self.info_view = self._info_label()
        v4.addWidget(self.info_view)
        v4.addStretch(1)

        for i, c in enumerate((c1, c2, c3, c4)):
            if i:
                chain.addWidget(self._arrow())
            chain.addWidget(c, 1)

        # 5 · Renderers
        self.box_render = c5 = QtWidgets.QGroupBox()
        v5 = QtWidgets.QVBoxLayout(c5)
        r5 = QtWidgets.QHBoxLayout()
        r5.addWidget(QtWidgets.QLabel("Primitive:"))
        self.cmb_shape2 = QtWidgets.QComboBox()
        self.cmb_shape2.addItems(PRIMITIVES)
        r5.addWidget(self.cmb_shape2)
        btn_cam = QtWidgets.QPushButton("Reset camera")
        btn_cam.clicked.connect(self.camera.reset)
        r5.addWidget(btn_cam)
        r5.addStretch(1)
        v5.addLayout(r5)
        row, self.pipe_srgb, self.pipe_acescg = self._viewport_pair()
        v5.addLayout(row, 1)
        outer.addWidget(c5, 1)
        return page

    def _wire(self) -> None:
        for w in (self.chk_srgb, self.chk_convert, self.chk_matrix):
            w.toggled.connect(self.schedule_texture)
        self.chk_convert.toggled.connect(self._sync_enabled)
        self.chk_matrix.toggled.connect(self._sync_enabled)

        # Pipeline-tab dropdowns <-> Viewer-tab controls (index of the "checked" state)
        self._link_combo_check(self.cmb_interp, self.chk_srgb, checked_index=0)
        self._link_combo_check(self.cmb_conv, self.chk_convert, checked_index=1)
        self._link_combo_check(self.cmb_mat, self.chk_matrix, checked_index=1)
        self._link_combos(self.cmb_view, self.cmb_view2)
        self._link_combos(self.cmb_shape, self.cmb_shape2)

        for cb in (self.cmb_view, self.cmb_view2):
            cb.currentIndexChanged.connect(lambda _=0: (self.request_redraw(), self.request_previews()))
        for cb in (self.cmb_shape, self.cmb_shape2):
            cb.currentIndexChanged.connect(lambda _=0: (self.request_redraw(), self.request_previews()))
        for cb in self.show_combos.values():
            cb.currentIndexChanged.connect(lambda _=0: self.request_previews())
        self.cmb_preview.currentIndexChanged.connect(lambda _=0: self.request_previews())
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.camera.changed.connect(lambda: self.request_redraw(fast=True))
        self.settings.changed.connect(self.on_settings_changed)

    @staticmethod
    def _link_combo_check(combo: QtWidgets.QComboBox, chk: QtWidgets.QCheckBox, checked_index: int) -> None:
        def from_combo(i: int) -> None:
            want = i == checked_index
            if chk.isChecked() != want:
                chk.setChecked(want)

        def from_chk(v: bool) -> None:
            want = checked_index if v else 1 - checked_index
            if combo.currentIndex() != want:
                combo.setCurrentIndex(want)

        combo.currentIndexChanged.connect(from_combo)
        chk.toggled.connect(from_chk)
        from_chk(chk.isChecked())

    @staticmethod
    def _link_combos(a: QtWidgets.QComboBox, b: QtWidgets.QComboBox) -> None:
        a.currentTextChanged.connect(lambda t: b.setCurrentText(t) if b.currentText() != t else None)
        b.currentTextChanged.connect(lambda t: a.setCurrentText(t) if a.currentText() != t else None)

    # ------------------------------------------------------ Settings tab --
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Changes apply immediately and are saved between sessions."))
        top.addStretch(1)
        btn = QtWidgets.QPushButton("Reset all to defaults")
        btn.clicked.connect(self.settings.reset)
        top.addWidget(btn)
        outer.addLayout(top)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(inner)
        for title, items in SETTING_GROUPS:
            box = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(box)
            for name, label, kind, lo, hi, step, tip in items:
                w = self._make_setting_widget(name, kind, lo, hi, step)
                if tip:
                    w.setToolTip(tip)
                form.addRow(label + ":", w)
            col.addWidget(box)
        col.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        return page

    def _make_setting_widget(self, name: str, kind: str, lo: float, hi: float, step: float) -> QtWidgets.QWidget:
        value = getattr(self.d, name)
        put = lambda v: self.settings.set(name, v)  # noqa: E731

        def dspin() -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(4 if step < 0.01 else 3)
            return sb

        if kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(step))
            w.setValue(int(value))
            w.valueChanged.connect(lambda v: put(int(v)))
            self.setting_setters[name] = lambda v, w=w: (w.blockSignals(True), w.setValue(int(v)), w.blockSignals(False))
        elif kind == "float":
            w = dspin()
            w.setValue(float(value))
            w.valueChanged.connect(lambda v: put(float(v)))
            self.setting_setters[name] = lambda v, w=w: (w.blockSignals(True), w.setValue(float(v)), w.blockSignals(False))
        elif kind == "vec3":
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            sbs = [dspin() for _ in range(3)]
            for sb, v in zip(sbs, value):
                sb.setValue(float(v))
                h.addWidget(sb)
                sb.valueChanged.connect(lambda _=0: put(tuple(x.value() for x in sbs)))

            def set_vec(v, sbs=sbs) -> None:
                for sb, x in zip(sbs, v):
                    sb.blockSignals(True)
                    sb.setValue(float(x))
                    sb.blockSignals(False)
            self.setting_setters[name] = set_vec
        elif kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v: put(bool(v)))
            self.setting_setters[name] = lambda v, w=w: (w.blockSignals(True), w.setChecked(bool(v)), w.blockSignals(False))
        elif kind == "str":
            w = QtWidgets.QLineEdit(str(value))
            w.editingFinished.connect(lambda w=w: put(w.text().strip()))
            self.setting_setters[name] = lambda v, w=w: w.setText(str(v))
        elif kind.startswith("choice:"):
            opts = kind.split(":", 1)[1].split(",")
            w = QtWidgets.QComboBox()
            w.addItems(opts)
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda t: put(int(t)))
            self.setting_setters[name] = lambda v, w=w: (w.blockSignals(True), w.setCurrentText(str(v)), w.blockSignals(False))
        else:  # pragma: no cover
            raise ValueError(kind)
        return w

    def on_settings_changed(self, name: str) -> None:
        d = self.d
        if name == "*":  # full reset: refresh every widget
            for n, setter in self.setting_setters.items():
                setter(getattr(d, n))
        self.tex_timer.setInterval(d.texture_debounce_ms)
        self.refine_timer.setInterval(d.refine_delay_ms)
        self.thumb_cam.reset(emit=False)
        for lb in self.thumb_labels:
            lb.setFixedSize(d.thumb_width, d.thumb_height)
        if name in ("thumb_source_size", "*") and self.image is not None:
            self.thumb = box_downscale(self.image, d.thumb_source_size)
        self.scene_709 = scene_from_settings(d)
        self.scene_acescg = self.ctx.convert(self.scene_709.reshape(1, 4, 3), self.ctx.lin709,
                                             self.ctx.acescg).reshape(4, 3)
        self.gl.settings_changed(name)
        if name in ("anisotropy", "*") and self.tex is not None:
            self.gl.set_texture(self.tex)  # re-upload with the new filtering
        self.request_redraw()
        self.request_previews()

    # -------------------------------------------- matrix paste (clipboard) --
    def paste_matrix(self) -> None:
        text = QtWidgets.QApplication.clipboard().text()
        try:
            m = parse_matrix(text, column_major=self.d.paste_column_major)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self, "Insert matrix",
                f"{e}\n\nSupported: spaced columns with one row per line, or brackets and commas, e.g.\n"
                "0.6 0.3 0.1\n0.1 0.8 0.1\n0.05 0.1 0.85\n\n[[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]]")
            return
        for sb, val in zip(self.spins, m.flatten()):
            sb.blockSignals(True)
            sb.setValue(float(val))
            sb.blockSignals(False)
        self.chk_convert.setChecked(True)   # the matrix only takes effect when converting
        self.chk_matrix.setChecked(True)
        self._sync_enabled()
        self.schedule_texture()
        self.request_previews()
        rows = "; ".join(" ".join(f"{v:g}" for v in r) for r in m)
        self.statusBar().showMessage(f"Matrix inserted from clipboard: {rows}", 8000)

    # ------------------------------------ "show this step in the renderers" --
    def on_override(self, key: str, checked: bool) -> None:
        if checked:
            for k, chk in self.override_checks.items():  # exclusive
                if k != key and chk.isChecked():
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)
            self.override_key = key
        elif self.override_key == key:
            self.override_key = None
        self.apply_render_texture()

    def render_texture_array(self) -> np.ndarray | None:
        st = self.stages_full
        if st is None:
            return None
        pick = {"in": st.inp, "ws": st.ws, "mx": st.mx, "view": st.mx}
        return np.nan_to_num(pick.get(self.override_key or "mx", st.mx))

    def apply_render_texture(self) -> None:
        arr = self.render_texture_array()
        if arr is not None:
            self.tex = arr
            self.gl.set_texture(arr)
        self.update_captions()
        self.redraw(False)

    def update_captions(self) -> None:
        ctx = self.ctx
        suffix = f"  ·  step: {self.STAGE_TITLES[self.override_key]}" if self.override_key else ""
        for cap in self.caps_srgb:
            cap.setText(f"sRGB renderer  ({ctx.lin709}){suffix}")
        for cap in self.caps_acescg:
            cap.setText(f"ACEScg renderer  ({ctx.acescg}){suffix}")
        src = f"step '{self.STAGE_TITLES[self.override_key]}'" if self.override_key else "final texture"
        self.box_render.setTitle(f"5 · Renderers  ({src} → primitive → view transform)")

    def on_tab_changed(self, _i: int) -> None:
        self.request_redraw()
        self.request_previews()

    # ------------------------------------------------- context / config --
    def bind_context(self, ctx: OcioContext) -> None:
        self.ctx = ctx
        self.ocio_matrix = ctx.gamut_matrix()
        self.scene_709 = scene_from_settings(self.d)
        self.scene_acescg = ctx.convert(self.scene_709.reshape(1, 4, 3), ctx.lin709, ctx.acescg).reshape(4, 3)
        for cb in (self.cmb_view, self.cmb_view2):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(ctx.views)
            cb.setCurrentText(ctx.default_view)
            cb.blockSignals(False)
        for lb in (self.lbl_display, self.lbl_display2):
            lb.setText(ctx.display)
        self.lbl_cfg.setText(f"Config: {ctx.config.getName() or ctx.config.getDescription()[:60]}")
        self.update_captions()
        self.reset_matrix()
        self._sync_enabled()
        self.schedule_texture()
        self.request_previews()

    def load_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "OCIO config", "", "OCIO config (*.ocio)")
        if not path:
            return
        try:
            self.bind_context(OcioContext(OCIO.Config.CreateFromFile(path)))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Config error", str(e))

    # ------------------------------------------------------------ matrix --
    def reset_matrix(self) -> None:
        for sb, val in zip(self.spins, self.ocio_matrix.flatten()):
            sb.blockSignals(True)
            sb.setValue(float(val))
            sb.blockSignals(False)
        self.schedule_texture()
        self.request_previews()

    def matrix(self) -> np.ndarray:
        return np.array([sb.value() for sb in self.spins], dtype=np.float32).reshape(3, 3)

    def _sync_enabled(self) -> None:
        conv = self.chk_convert.isChecked()
        self.chk_matrix.setEnabled(conv)
        self.cmb_mat.setEnabled(conv)
        on = conv and self.chk_matrix.isChecked()
        for sb in self.spins:
            sb.setEnabled(on)

    # ----------------------------------------------------------- loading --
    def load_path(self, path: str) -> None:
        try:
            arr, is_float = load_image(path, self.d.max_texture_size)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return
        self.image, self.image_path = arr, path
        self.thumb = box_downscale(arr, self.d.thumb_source_size)
        self.chk_srgb.setChecked(not is_float)  # EXR/HDR are normally linear
        prev = np.clip(arr, 0, 1)
        if is_float:
            prev = prev ** (1 / 2.2)
        self.slot.set_preview(QtGui.QPixmap.fromImage(to_qimage((prev * 255 + 0.5).astype(np.uint8))))
        self.schedule_texture()
        self.request_previews()

    # -------------------------------------------- pipeline: final texture --
    def schedule_texture(self) -> None:
        self.tex_timer.start()

    def _params(self) -> tuple[bool, bool, bool]:
        conv = self.chk_convert.isChecked()
        return self.chk_srgb.isChecked(), conv, conv and self.chk_matrix.isChecked()

    def describe_pipeline(self) -> list[str]:
        ctx = self.ctx
        is_srgb, conv, override = self._params()
        h, w = self.image.shape[:2]  # type: ignore[union-attr]
        log = [f"Input  : {Path(self.image_path).name}  {w}x{h}  "
               f"interpreted as {'sRGB-encoded' if is_srgb else 'linear/raw'}"]
        if not conv:
            log.append("Convert OFF -> texture = pixel values as loaded (no OCIO applied)")
            return log
        log.append(f"Decode : '{ctx.srgb_tex}' -> '{ctx.lin709}'" if is_srgb
                   else f"Decode : none (raw treated as '{ctx.lin709}')")
        if override:
            log.append("Gamut  : custom matrix override\n         " + "\n         ".join(
                " ".join(f"{v: .5f}" for v in r) for r in self.matrix()))
        else:
            log.append(f"Gamut  : OCIO '{ctx.lin709}' -> '{ctx.acescg}'")
        return log

    def rebuild_texture(self) -> None:
        if self.image is None:
            return
        try:
            is_srgb, conv, override = self._params()
            st = run_pipeline(self.ctx, self.image, is_srgb, conv, override, self.matrix())
            self.stages_full = st
            self.pipeline_log = self.describe_pipeline() + [f"Final texture {fmt_rgb(st.mx)}"]
            self.tex = self.render_texture_array()   # final texture, or the step chosen for override
            self.gl.set_texture(self.tex)
        except Exception as e:  # noqa: BLE001
            self.pipeline_log = [f"ERROR: {e}"]
            self.tex = None
        self.redraw(False)
        self.request_previews()

    # ------------------------------------- rendering (GPU + OCIO display) --
    def active_viewports(self) -> tuple[ViewportWidget, ViewportWidget] | None:
        i = self.tabs.currentIndex()
        if i == 0:
            return self.view_srgb, self.view_acescg
        if i == 1:
            return self.pipe_srgb, self.pipe_acescg
        return None  # Settings tab: nothing to draw

    def request_redraw(self, fast: bool = False) -> None:
        self._fast = self._fast or fast
        self.draw_timer.start()
        if fast:
            self.refine_timer.start()  # full-resolution pass when interaction stops

    def redraw(self, fast: bool) -> None:
        self._fast = False
        self.update_log()
        if self.tex is None:
            return
        ctx, view, shape = self.ctx, self.cmb_view.currentText(), self.cmb_shape.currentText()
        active = self.active_viewports()
        if active is None:
            return
        vp_a, vp_b = active
        try:
            for ws, scene, vp in ((ctx.lin709, self.scene_709, vp_a), (ctx.acescg, self.scene_acescg, vp_b)):
                W, H = vp.pixel_size()
                s = min(1.0, self.d.max_render_size / max(W, H)) * (self.d.interactive_scale if fast else 1.0)
                w, h = max(16, int(W * s)), max(16, int(H * s))
                lin = self.gl.render(shape, self.camera, scene, w, h)
                disp = ctx.to_display(np.maximum(lin, 0.0), ws, view)
                vp.set_image(to_qimage((np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)))
        except Exception as e:  # noqa: BLE001
            self.log.setPlainText("\n".join(self.pipeline_log + [f"RENDER ERROR: {e}"]))

    def update_log(self) -> None:
        if not self.pipeline_log or self.image is None:
            return
        shown = f"step '{self.STAGE_TITLES[self.override_key]}' (override)" if self.override_key else "final texture"
        extra = [f"Renderers show: {shown}",
                 f"Shape  : {self.cmb_shape.currentText()}   (GPU: {self.gl.info})",
                 f"View   : working space -> display '{self.ctx.display}' / view '{self.cmb_view.currentText()}'",
                 "Both viewports use the SAME texture numbers and the SAME lights "
                 "(lights converted to each working space)."]
        self.log.setPlainText("\n".join(self.pipeline_log + extra))

    # ------------------------------------------- Pipeline tab: thumbnails --
    def request_previews(self) -> None:
        self.prev_timer.start()

    def _set_thumb(self, label: QtWidgets.QLabel, arr: np.ndarray) -> None:
        u8 = (np.clip(np.nan_to_num(arr), 0, 1) * 255 + 0.5).astype(np.uint8)
        pix = QtGui.QPixmap.fromImage(to_qimage(u8))
        label.setPixmap(pix.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                   QtCore.Qt.TransformationMode.SmoothTransformation))

    def update_previews(self) -> None:
        if self.tabs.currentIndex() != 1 or self.thumb is None or self.image is None:
            return
        ctx, view = self.ctx, self.cmb_view.currentText()
        shape = self.cmb_shape.currentText()
        is_srgb, conv, override = self._params()
        notes = {k: "" for k in self.show_combos}
        try:
            st = run_pipeline(ctx, self.thumb, is_srgb, conv, override, self.matrix())
            managed = self.cmb_preview.currentIndex() == 1
            stages = {"in": (st.inp, st.inp_cs, self.th_in), "ws": (st.ws, st.ws_cs, self.th_ws),
                      "mx": (st.mx, st.mx_cs, self.th_mx), "view": (st.mx, st.mx_cs, self.th_view)}
            for key, (arr, cs, label) in stages.items():
                arr = np.nan_to_num(arr)
                mode = self.show_combos[key].currentIndex()
                if mode == 0:  # flat texture
                    if managed or key == "view":
                        disp = ctx.to_display(np.maximum(arr, 0), cs, view)
                    else:
                        disp = arr
                else:  # render this stage's texture on the primitive with the chosen renderer
                    ws, scene = (ctx.lin709, self.scene_709) if mode == 1 else (ctx.acescg, self.scene_acescg)
                    tid = self.gl.set_stage_texture(key, arr)
                    tw, th = self.d.thumb_render_width, self.d.thumb_render_height
                    lin = self.gl.render(shape, self.thumb_cam, scene, tw, th, texture=tid,
                                         aspect=arr.shape[1] / arr.shape[0])
                    disp = ctx.to_display(np.maximum(lin, 0), ws, view)
                    notes[key] = (f"\nRendered on {shape} by the {self.SHOW_MODES[mode]}: numbers read as "
                                  f"'{ws}', then view '{view}'.")
                self._set_thumb(label, disp)
        except Exception as e:  # noqa: BLE001
            self.info_view.setText(f"Preview error: {e}")
            return

        h, w = self.image.shape[:2]
        self.info_in.setText(f"{Path(self.image_path).name}\n{w}x{h} · read as '{st.inp_cs}'\n"
                             f"{fmt_rgb(st.inp)}{notes['in']}")
        if conv:
            chain = (f"'{ctx.srgb_tex}' → " if is_srgb else "") + f"'{ctx.lin709}' → '{ctx.acescg}' (OCIO)"
            self.info_ws.setText(f"{chain}\n{fmt_rgb(st.ws)}{notes['ws']}")
        else:
            self.info_ws.setText(f"No conversion: values kept as loaded and read as '{ctx.lin709}'.\n"
                                 f"{fmt_rgb(st.ws)}{notes['ws']}")
        if not conv:
            self.info_mx.setText(f"n/a (conversion is off){notes['mx']}")
        else:
            m, tag = (self.matrix(), "custom") if override else (self.ocio_matrix, "OCIO default")
            rows = "\n".join(" ".join(f"{v: .4f}" for v in r) for r in m)
            self.info_mx.setText(f"{tag}\n{rows}\n{fmt_rgb(st.mx)}{notes['mx']}")
        self.info_view.setText(f"'{st.mx_cs}' → display '{ctx.display}', view '{view}'.{notes['view']}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("OCIOPipelineViewer")
    settings = Settings()
    try:
        cfg = OCIO.Config.CreateFromEnv() if os.environ.get("OCIO") \
            else OCIO.Config.CreateFromFile(settings.d.default_config)
        ctx = OcioContext(cfg)
    except Exception as e:  # noqa: BLE001
        QtWidgets.QMessageBox.critical(None, "OCIO error", f"Could not initialise OCIO config:\n{e}")
        return 1
    try:
        renderer = GLRenderer(settings.d)
    except Exception as e:  # noqa: BLE001
        QtWidgets.QMessageBox.critical(None, "OpenGL error", f"Could not initialise OpenGL 3.3:\n{e}")
        return 1
    win = MainWindow(ctx, renderer, settings)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())