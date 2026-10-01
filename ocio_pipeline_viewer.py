#!/usr/bin/env python3
"""
OCIO Pipeline Viewer

How rendering works: OpenGL shades in floating point in the renderer's working space
(MSAA float framebuffer), the result is read back and OCIO applies the selected
display/view transform on the CPU. While you drag, it renders at half resolution and
refines to full resolution when you stop.

View-transform override: the Viewer tab has an HLSL editor. When "Override" is ticked the
shader replaces the OCIO view transform in every rendered preview. Contract:
    float3 ViewTransform(float3 color, float3x3 inputMatrix)
HLSL is translated to GLSL 3.30 and compiled on the GPU (errors keep your line numbers).
"Reset to default" restores the built-in shader.

Colour spaces: the input space and the working space are picked from cascading menus built from the
loaded OCIO config: Roles, then the Family tree ("Input/Camera" -> Input > Camera), then
Categories, separated by lines. The conversion is the OCIO transform between the two spaces.

Settings: saved as ocio_pipeline_viewer_settings.json next to the .exe (or the script); if that
folder is read-only, in the user's config folder. A custom start-up OCIO config that fails to load
is reset to the built-in config. Relative config paths are relative to the .exe folder.
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


BUILTIN_CONFIG = "ocio://cg-config-latest"      # built-in config (OCIO >= 2.3)
SETTINGS_FILENAME = "ocio_pipeline_viewer_settings.json"
NO_CONVERSION = "None (values as-is)"


def app_dir() -> Path:
    """Folder of the .exe when frozen/compiled (PyInstaller, Nuitka, ...), else of this script."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent


def resolve_config_ref(ref: str) -> str:
    """`ocio://...` URIs are kept; relative file paths are relative to the .exe folder."""
    ref = ref.strip()
    if ref.startswith("ocio://"):
        return ref
    p = Path(os.path.expandvars(ref)).expanduser()
    return str(p if p.is_absolute() else app_dir() / p)


def portable_ref(path: str) -> str:
    """Store a config path relative to the .exe folder when possible (keeps the install portable)."""
    if path.startswith("ocio://"):
        return path
    try:
        return os.path.relpath(path, app_dir())
    except ValueError:  # different drive on Windows
        return path


def default_settings_path() -> Path:
    """Settings live next to the .exe; if that folder is read-only, in the user's config folder."""
    if os.access(app_dir(), os.W_OK):
        return app_dir() / SETTINGS_FILENAME
    base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppConfigLocation)
    folder = Path(base) if base else Path.home()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / SETTINGS_FILENAME


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
        # (name, family, categories) for every colour space in the config -> drives the tree pickers
        self.cs_meta: list[tuple[str, str, list[str]]] = []
        for cs in config.getColorSpaces():
            self.cs_meta.append((cs.getName(), cs.getFamily() or "", [str(c) for c in cs.getCategories()]))
        for required in (self.srgb_tex, self.lin709, self.acescg):
            if required not in {m[0] for m in self.cs_meta}:
                cs = config.getColorSpace(required)
                self.cs_meta.append((required, cs.getFamily() or "", [str(c) for c in cs.getCategories()]))
        self.colorspaces = [m[0] for m in self.cs_meta]

        self.family_sep = "/"
        try:
            sep = config.getFamilySeparator()
            if sep and ord(sep[0]) > 0:
                self.family_sep = sep[0]
        except Exception:  # noqa: BLE001
            pass

        self.roles: list[tuple[str, str]] = []   # (role, colour space name)
        try:
            known = set(self.colorspaces)
            for item in config.getRoles():
                role, name = item if isinstance(item, (tuple, list)) else (item, config.getRoleColorSpace(item))
                if name in known:
                    self.roles.append((str(role), str(name)))
        except Exception:  # noqa: BLE001
            pass
        self.roles.sort(key=lambda r: r[0].lower())
        self.source = ""  # where the config was loaded from (for display)

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

    def linear_matrix(self, src: str, dst: str) -> np.ndarray:
        """3x3 matrix src -> dst as OCIO computes it (out = M @ rgb). Exact for linear colour spaces;
        for non-linear ones it is only the linearised approximation."""
        basis = np.eye(3, dtype=np.float32).reshape(1, 3, 3)  # three RGB pixels
        cols = self.convert(basis, src, dst).reshape(3, 3)    # row i = M @ e_i
        return cols.T.astype(np.float64)

    def gamut_matrix(self, target: str | None = None) -> np.ndarray:
        """Linear Rec.709 -> `target` (default ACEScg)."""
        return self.linear_matrix(self.lin709, target or self.acescg)


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


DEFAULT_VIEW_SHADER = """// ---- Custom view transform (HLSL) -------------------------------------------------
// Replaces the OCIO view transform in all previews while "Override with custom
// shader" is ticked. Runs per pixel on the rendered image, in the renderer's
// working space.
//
//   color             linear scene RGB in the renderer's working space
//                     (Linear Rec.709 for the sRGB renderer, ACEScg for the ACEScg one)
//   inputMatrix       the matrix of step 3 (your custom values, else the OCIO default).
//                     Row-major like HLSL:  mul(inputMatrix, color)
//   workingToRec709   global float3x3: working-space primaries -> Rec.709 primaries
//   returns           display-referred, sRGB-encoded RGB (0..1)
//
// Supported: float / float2 / float3 / float4, float3x3, float4x4, mul, saturate,
// lerp, frac, fmod, mad, rsqrt, atan2, log10, ddx/ddy, static, #define / #if.
// Not supported: implicit scalar -> vector promotion (write float3(0.5), not 0.5).

#define USE_INPUT_MATRIX 0      // 1 = multiply the colour by inputMatrix first
#define TONEMAP          0      // 1 = simple Reinhard tone-map

float3 SrgbOetf(float3 c)
{
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return lerp(hi, lo, step(c, float3(0.0031308)));
}

float3 ViewTransform(float3 color, float3x3 inputMatrix)
{
    float3 c = max(color, float3(0.0));
#if USE_INPUT_MATRIX
    c = mul(inputMatrix, c);
#endif
    c = mul(workingToRec709, c);            // to the display primaries
    c = max(c, float3(0.0));
#if TONEMAP
    c = c / (1.0 + c);
#endif
    return SrgbOetf(saturate(c));
}
"""


# --------------------------------------------------------------------------- #
# Settings (all tunable "magic numbers"; edited in the Settings tab, saved as JSON next to the .exe)
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
    default_config: str = BUILTIN_CONFIG
    view_shader: str = DEFAULT_VIEW_SHADER   # edited in the Viewer tab, not listed in the Settings tab


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
         "Used when $OCIO is not set. ocio://... or a .ocio path (relative paths are relative to the .exe). "
         "If it fails to load, it is reset to the built-in config."),
    ]),
]


class Settings(QtCore.QObject):
    """Holds SettingsData (`.d`, a stable object edited in place) and persists it as a JSON file."""
    changed = QtCore.Signal(str)  # field name, or "*" for a full reset

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.d = SettingsData()
        self.path = Path(path) if path else default_settings_path()
        self.save_error = ""
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
        except (OSError, ValueError):
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
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(self.d), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # atomic: never leaves a half-written file
            self.save_error = ""
        except OSError as e:
            self.save_error = f"Could not save settings to {self.path}: {e}"

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

def _text(value) -> str:
    """PyOpenGL returns GL strings / info logs as bytes or str depending on version and platform."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode(errors="replace")
    return str(value)


def _compile(kind: int, src: str) -> int:
    sh = gl.glCreateShader(kind)
    gl.glShaderSource(sh, src)
    gl.glCompileShader(sh)
    if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
        raise RuntimeError("Shader compile error:\n" + _text(gl.glGetShaderInfoLog(sh)))
    return sh


# --------------------------------------------------------------------------- #
# HLSL-flavoured post shader (custom view transform)
#   HLSL source -> GLSL 3.30: types/intrinsics via macros + overloads, so line numbers in compiler
#   messages match the editor. Matrices are uploaded with HLSL semantics: GLSL M[i] is HLSL row i,
#   so mul(M, v) == M @ v in numpy.
# --------------------------------------------------------------------------- #
POST_VERT_SRC = """
#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

HLSL_PRELUDE = """
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define half float
#define half2 vec2
#define half3 vec3
#define half4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define bool2 bvec2
#define bool3 bvec3
#define bool4 bvec4
#define float2x2 mat2
#define float3x3 mat3
#define float4x4 mat4
#define static
#define lerp mix
#define frac fract
#define rsqrt inversesqrt
#define atan2 atan
#define ddx dFdx
#define ddy dFdy
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2 saturate(vec2 x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }
vec4 saturate(vec4 x) { return clamp(x, 0.0, 1.0); }
float mad(float a, float b, float c) { return a * b + c; }
vec3 mad(vec3 a, vec3 b, vec3 c) { return a * b + c; }
float fmod(float x, float y) { return x - y * trunc(x / y); }
vec3 fmod(vec3 x, vec3 y) { return x - y * trunc(x / y); }
float log10(float x) { return log(x) * 0.43429448190325176; }
vec3 log10(vec3 x) { return log(x) * 0.43429448190325176; }
vec2 pow(vec2 x, float y) { return pow(x, vec2(y)); }
vec3 pow(vec3 x, float y) { return pow(x, vec3(y)); }
vec4 pow(vec4 x, float y) { return pow(x, vec4(y)); }
vec3 mul(mat3 m, vec3 v) { return v * m; }
vec3 mul(vec3 v, mat3 m) { return m * v; }
mat3 mul(mat3 a, mat3 b) { return b * a; }
vec4 mul(mat4 m, vec4 v) { return v * m; }
vec4 mul(vec4 v, mat4 m) { return m * v; }
mat4 mul(mat4 a, mat4 b) { return b * a; }
vec2 mul(mat2 m, vec2 v) { return v * m; }
vec2 mul(vec2 v, mat2 m) { return m * v; }
"""


def hlsl_to_glsl(src: str) -> str:
    """Light source fix-ups (keeps the number of lines unchanged)."""
    return re.sub(r"\[\s*(?:unroll|loop|branch|flatten|fastopt|allow_uav_condition)[^\]]*\]", "", src)


def build_post_fragment(user_src: str) -> str:
    return "\n".join([
        "#version 330 core",
        HLSL_PRELUDE,
        "uniform sampler2D uImage;",
        "uniform mat3 uInputMatrix;",
        "uniform mat3 workingToRec709;",
        "in vec2 vUV;",
        "out vec4 outColor;",
        "#line 1",
        hlsl_to_glsl(user_src),
        "",
        "void main() {",
        "    vec3 c = texture(uImage, vUV).rgb;",
        "    outColor = vec4(ViewTransform(c, uInputMatrix), 1.0);",
        "}",
    ])


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

        self.info = _text(gl.glGetString(gl.GL_RENDERER)) or "?"
        vs, fs = _compile(gl.GL_VERTEX_SHADER, VERT_SRC), _compile(gl.GL_FRAGMENT_SHADER, FRAG_SRC)
        self.prog = gl.glCreateProgram()
        gl.glAttachShader(self.prog, vs)
        gl.glAttachShader(self.prog, fs)
        gl.glLinkProgram(self.prog)
        if not gl.glGetProgramiv(self.prog, gl.GL_LINK_STATUS):
            raise RuntimeError("Program link error:\n" + _text(gl.glGetProgramInfoLog(self.prog)))

        self.texture = int(gl.glGenTextures(1))
        self.stage_tex: dict[str, int] = {}
        # custom view-transform (post) pass
        self._post_vs = _compile(gl.GL_VERTEX_SHADER, POST_VERT_SRC)
        self.post_prog = 0
        self.post_fbos: dict[tuple[int, int], dict[str, int]] = {}
        self.empty_vao = int(gl.glGenVertexArrays(1))
        self.flat_tex = int(gl.glGenTextures(1))
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

    # ---- custom view-transform shader ----
    def compile_post(self, user_src: str) -> tuple[bool, str]:
        """Compile the HLSL view-transform source. The previous program is kept until a new one links."""
        self.make_current()
        try:
            fs = _compile(gl.GL_FRAGMENT_SHADER, build_post_fragment(user_src))
        except RuntimeError as e:
            return False, str(e).replace("Shader compile error:\n", "")
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, self._post_vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        ok = bool(gl.glGetProgramiv(prog, gl.GL_LINK_STATUS))
        log = _text(gl.glGetProgramInfoLog(prog))
        gl.glDeleteShader(fs)
        if not ok:
            gl.glDeleteProgram(prog)
            return False, "Link error:\n" + log
        if self.post_prog:
            gl.glDeleteProgram(self.post_prog)
        self.post_prog = prog
        return True, log

    def _ensure_post_fbo(self, w: int, h: int) -> dict[str, int]:
        key = (w, h)
        if key in self.post_fbos:
            return self.post_fbos[key]
        while len(self.post_fbos) >= 8:
            old = self.post_fbos.pop(next(iter(self.post_fbos)))
            gl.glDeleteFramebuffers(1, [old["fbo"]])
            gl.glDeleteTextures(1, [old["tex"]])
        tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        fbo = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("post-pass framebuffer incomplete")
        self.post_fbos[key] = {"fbo": fbo, "tex": tex}
        return self.post_fbos[key]

    @staticmethod
    def _read_rgb(w: int, h: int) -> np.ndarray:
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_FLOAT)
        arr = np.frombuffer(data, dtype=np.float32) if isinstance(data, (bytes, bytearray)) \
            else np.asarray(data, dtype=np.float32)
        return arr.reshape(h, w, 3)

    def _run_post(self, src_tex: int, w: int, h: int, matrix: np.ndarray | None,
                  to709: np.ndarray | None) -> np.ndarray:
        """Run the compiled view shader over `src_tex`; returns rows in the texture's row order."""
        f = self._ensure_post_fbo(w, h)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f["fbo"])
        gl.glViewport(0, 0, w, h)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.post_prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self.post_prog, "uImage"), 0)
        for name, m in (("uInputMatrix", matrix), ("workingToRec709", to709)):
            m = np.eye(3) if m is None else m
            # row-major numpy data, no transpose: GLSL column i == matrix row i (HLSL semantics)
            gl.glUniformMatrix3fv(gl.glGetUniformLocation(self.post_prog, name), 1, gl.GL_FALSE,
                                  np.ascontiguousarray(m, dtype=np.float32).reshape(9))
        gl.glBindVertexArray(self.empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        out = self._read_rgb(w, h)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return out

    def apply_view_shader(self, arr: np.ndarray, matrix: np.ndarray | None, to709: np.ndarray | None) -> np.ndarray:
        """Run the view shader over a flat image (top-down float RGB); returns display-referred RGB."""
        if not self.post_prog:
            raise RuntimeError("no compiled view shader")
        self.make_current()
        h, w, _ = arr.shape
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.flat_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, w, h, 0, gl.GL_RGB, gl.GL_FLOAT, arr)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        return self._run_post(self.flat_tex, w, h, matrix, to709)

    # ---- draw ----
    def render(self, shape: str, cam: Camera, scene: np.ndarray, w: int, h: int,
               texture: int | None = None, aspect: float | None = None, post: bool = False,
               input_matrix: np.ndarray | None = None, to709: np.ndarray | None = None) -> np.ndarray:
        """Returns float32 RGB (top-down). Working-space values, or - with `post` - the output of the
        compiled view shader (display-referred), run with `input_matrix` / `to709`."""
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
        if post and self.post_prog:
            return self._run_post(fb["tex"], w, h, input_matrix, to709)[::-1]  # GL is bottom-up
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fb["resolve"])
        arr = self._read_rgb(w, h)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return arr[::-1]  # GL is bottom-up


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
# Colour-space picker: cascading menu (Photoshop / 3ds Max style) built from the OCIO config
#   [extra entries] ---- Roles > ... ---- Families (tree) ---- Categories > ...
# --------------------------------------------------------------------------- #
class _FamilyNode:
    __slots__ = ("kids", "items")

    def __init__(self) -> None:
        self.kids: dict[str, _FamilyNode] = {}
        self.items: list[str] = []


def populate_cs_menu(menu: QtWidgets.QMenu, ctx: OcioContext, pick, extras: list[str]) -> None:
    menu.clear()

    def add(m: QtWidgets.QMenu, label: str, name: str) -> None:
        act = m.addAction(label)
        act.triggered.connect(lambda _checked=False, n=name: pick(n))

    for e in extras:
        add(menu, e, e)
    if extras:
        menu.addSeparator()

    if ctx.roles:                                     # roles: their own sub-menu
        rm = menu.addMenu("Roles")
        for role, cs in ctx.roles:
            add(rm, f"{role}  →  {cs}", cs)
        menu.addSeparator()

    root = _FamilyNode()                              # families: a tree ("Input/Camera" -> Input > Camera)
    for name, family, _cats in ctx.cs_meta:
        node = root
        parts = [p.strip() for p in family.split(ctx.family_sep) if p.strip()] if family else []
        for part in parts:
            node = node.kids.setdefault(part, _FamilyNode())
        node.items.append(name)

    def emit(m: QtWidgets.QMenu, node: _FamilyNode) -> None:
        for key in sorted(node.kids, key=str.lower):
            emit(m.addMenu(key), node.kids[key])
        for name in node.items:
            add(m, name, name)

    emit(menu, root)

    cats: dict[str, list[str]] = {}                   # categories, after all family classes
    for name, _family, cs_cats in ctx.cs_meta:
        for c in cs_cats:
            cats.setdefault(c, []).append(name)
    if cats:
        menu.addSeparator()
        header = menu.addAction("Categories")
        header.setEnabled(False)
        for c in sorted(cats, key=str.lower):
            sm = menu.addMenu(c)
            for name in cats[c]:
                add(sm, name, name)


class CsPicker(QtWidgets.QPushButton):
    """Button that opens the cascading colour-space menu. Mimics the bits of QComboBox used here
    (currentText, setCurrentText, currentTextChanged, currentIndexChanged)."""
    currentTextChanged = QtCore.Signal(str)
    currentIndexChanged = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__("")
        self._text = ""
        self._valid: set[str] = set()
        self._menu = QtWidgets.QMenu(self)
        self._menu.setStyleSheet("QMenu { menu-scrollable: 1; }")  # long sub-menus scroll
        self.setMenu(self._menu)
        self.setStyleSheet("QPushButton { text-align: left; padding: 3px 8px; }")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QtCore.QSize:
        s = super().sizeHint()
        s.setWidth(min(s.width(), 260))
        return s

    def minimumSizeHint(self) -> QtCore.QSize:
        s = super().minimumSizeHint()
        s.setWidth(140)
        return s

    def currentText(self) -> str:
        return self._text

    def setCurrentText(self, text: str) -> None:
        if text == self._text or text not in self._valid:
            return
        self._show(text)
        self.currentTextChanged.emit(text)
        self.currentIndexChanged.emit(0)

    def _show(self, text: str) -> None:
        self._text = text
        self.setText(text)
        self.setToolTip(text)

    def set_config(self, ctx: OcioContext, extras: tuple[str, ...] = (), keep: str = "", default: str = "") -> None:
        """(Re)build the menu from `ctx`; keeps `keep` if it still exists, else `default`. Silent."""
        self._valid = set(ctx.colorspaces) | set(extras)
        populate_cs_menu(self._menu, ctx, self.setCurrentText, list(extras))
        self._show(keep if keep in self._valid else default)


# --------------------------------------------------------------------------- #
# Pipeline stages (used for the final texture and for the Pipeline-tab thumbnails)
# --------------------------------------------------------------------------- #
@dataclass
class Stages:
    inp: np.ndarray      # 1. image as loaded
    inp_cs: str          #    colour space it is interpreted as
    ws: np.ndarray       # 2. after the OCIO conversion input space -> working space
    ws_cs: str
    mx: np.ndarray       # 3. after the (optional) matrix override; this is the final texture
    mx_cs: str


@dataclass
class PipelineParams:
    input_cs: str                 # colour space the image is stored in (from the OCIO config)
    target_cs: str | None         # working space to convert to; None = pass values through untouched
    override: bool                # use `matrix` instead of the OCIO transform
    matrix: np.ndarray            # 3x3, applied to the image decoded to Linear Rec.709


def run_pipeline(ctx: OcioContext, image: np.ndarray, p: PipelineParams) -> Stages:
    if p.target_cs is None:  # values are passed on untouched; read as Linear Rec.709 for display purposes
        return Stages(image, p.input_cs, image, ctx.lin709, image, ctx.lin709)
    ws = ctx.convert(image, p.input_cs, p.target_cs)              # OCIO: input space -> working space
    if p.override:                                                 # custom: decode to Rec.709 linear, then matrix
        lin = ctx.convert(image, p.input_cs, ctx.lin709)
        mx = (lin @ p.matrix.T).astype(np.float32)
    else:
        mx = ws
    return Stages(image, p.input_cs, ws, p.target_cs, mx, p.target_cs)


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
        self.shader_ok = False                       # custom view shader compiled successfully?
        self._to709_cache: dict[str, np.ndarray] = {}
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
    @staticmethod
    def _cs_combo() -> CsPicker:
        return CsPicker()

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

        left_widget = QtWidgets.QWidget()          # scrollable: the shader editor makes this column tall
        left = QtWidgets.QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 6, 0)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(460)
        lay.addWidget(left_scroll, 0)

        g1 = QtWidgets.QGroupBox("1 · Input image")
        l1 = QtWidgets.QVBoxLayout(g1)
        self.slot = DropSlot()
        self.slot.fileDropped.connect(self.load_path)
        self.cmb_input_cs = self._cs_combo()
        l1.addWidget(self.slot)
        f1v = QtWidgets.QFormLayout()
        f1v.addRow("Input colour space:", self.cmb_input_cs)
        l1.addLayout(f1v)
        left.addWidget(g1)

        g2 = QtWidgets.QGroupBox("2 · Working-space conversion")
        l2 = QtWidgets.QVBoxLayout(g2)
        self.cmb_target = self._cs_combo()
        f2v = QtWidgets.QFormLayout()
        f2v.addRow("Convert to:", self.cmb_target)
        l2.addLayout(f2v)
        note = QtWidgets.QLabel("Uses the OCIO transform from the input space to this space.\n"
                                "'None' = pixel values go to the renderers untouched.")
        note.setStyleSheet("color:#888;")
        l2.addWidget(note)
        left.addWidget(g2)

        g3 = QtWidgets.QGroupBox("4 · View transform")
        l3 = QtWidgets.QFormLayout(g3)
        self.cmb_view = QtWidgets.QComboBox()
        self.lbl_display = QtWidgets.QLabel()
        l3.addRow("Display:", self.lbl_display)
        l3.addRow("View:", self.cmb_view)
        self.chk_shader = QtWidgets.QCheckBox("Override with custom HLSL shader")
        self.chk_shader.setToolTip("Replaces the OCIO view transform in all previews with the shader below")
        l3.addRow(self.chk_shader)
        self.shader_panel = QtWidgets.QWidget()      # editor + buttons + log; shown only while the override is on
        sp = QtWidgets.QVBoxLayout(self.shader_panel)
        sp.setContentsMargins(0, 0, 0, 0)
        self.shader_edit = QtWidgets.QPlainTextEdit()
        self.shader_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.shader_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.shader_edit.setTabStopDistance(28)
        self.shader_edit.setMinimumHeight(260)
        self.shader_edit.setPlainText(self.d.view_shader)
        sp.addWidget(self.shader_edit)
        shader_btns = QtWidgets.QHBoxLayout()
        btn_compile = QtWidgets.QPushButton("Compile && apply")
        btn_compile.clicked.connect(self.compile_shader)
        btn_shader_reset = QtWidgets.QPushButton("Reset to default")
        btn_shader_reset.clicked.connect(self.reset_shader)
        shader_btns.addWidget(btn_compile)
        shader_btns.addWidget(btn_shader_reset)
        sp.addLayout(shader_btns)
        self.lbl_shader = QtWidgets.QLabel("Not compiled yet.")
        self.lbl_shader.setWordWrap(True)
        self.lbl_shader.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_shader.setStyleSheet("color:#888; font-size:11px;")
        sp.addWidget(self.lbl_shader)
        l3.addRow(self.shader_panel)
        self.shader_panel.setVisible(self.chk_shader.isChecked())   # hidden until the checkbox is ticked

        g4 = QtWidgets.QGroupBox("3 · Working-space conversion override")
        l4 = QtWidgets.QVBoxLayout(g4)
        self.chk_matrix = QtWidgets.QCheckBox("Use this matrix instead of the OCIO transform")
        l4.addWidget(self.chk_matrix)
        self.chk_shader_matrix = QtWidgets.QCheckBox("Use manual shader override")
        self.chk_shader_matrix.setToolTip(
            "Hand the manual matrix below to the view-transform shader (inputMatrix), even when the texture "
            "itself still uses the OCIO transform. Ticking it also switches the HLSL override on.")
        l4.addWidget(self.chk_shader_matrix)
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
        left.addWidget(g4)   # 3 · Input matrix
        left.addWidget(g3)   # 4 · View transform

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
        chk = QtWidgets.QCheckBox("Show this step in the final renderers")
        chk.setToolTip("Override: the final renderers (live Viewer tab + snapshots below) use this step's texture instead of the final one")
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
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("Primitive:"))
        self.cmb_shape2 = QtWidgets.QComboBox()
        self.cmb_shape2.addItems(PRIMITIVES)
        top.addWidget(self.cmb_shape2)
        top.addSpacing(16)
        top.addWidget(QtWidgets.QLabel("Stage snapshot camera:"))
        self.cmb_snapcam = QtWidgets.QComboBox()
        self.cmb_snapcam.addItems(["Default (from Settings)", "Same as live Viewer camera"])
        top.addWidget(self.cmb_snapcam)
        top.addStretch(1)
        outer.addLayout(top)

        chain = QtWidgets.QHBoxLayout()
        outer.addLayout(chain)

        # 1 · Input image
        c1, v1, self.th_in = self._card("1 · Input image", "in")
        self.cmb_interp = self._cs_combo()
        f1 = QtWidgets.QFormLayout()
        f1.addRow("Colour space:", self.cmb_interp)
        v1.addLayout(f1)
        self.info_in = self._info_label()
        v1.addWidget(self.info_in)
        v1.addStretch(1)

        # 2 · Working space
        c2, v2, self.th_ws = self._card("2 · Working space", "ws")
        self.cmb_conv = self._cs_combo()
        f2 = QtWidgets.QFormLayout()
        f2.addRow("Working space:", self.cmb_conv)
        v2.addLayout(f2)
        self.info_ws = self._info_label()
        v2.addWidget(self.info_ws)
        v2.addStretch(1)

        # 3 · Input matrix
        c3, v3, self.th_mx = self._card("3 · Input matrix", "mx")
        self.cmb_mat = QtWidgets.QComboBox()
        self.cmb_mat.addItems(["OCIO transform", "Custom matrix"])
        f3 = QtWidgets.QFormLayout()
        f3.addRow("Conversion:", self.cmb_mat)
        self.cmb_shmat = QtWidgets.QComboBox()
        self.cmb_shmat.addItems(["OCIO default matrix", "Manual shader override"])
        self.cmb_shmat.setToolTip("Matrix handed to the view-transform shader")
        f3.addRow("Shader gets:", self.cmb_shmat)
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

        # 5 · Final renderers: full live 3D (orbit / zoom / pan), same camera as the Viewer tab
        down = QtWidgets.QLabel("▼")
        down.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        down.setStyleSheet("font-size:20px; color:#888;")
        outer.addWidget(down)
        self.box_render = c5 = QtWidgets.QGroupBox()
        v5 = QtWidgets.QVBoxLayout(c5)
        r5 = QtWidgets.QHBoxLayout()
        btn_cam = QtWidgets.QPushButton("Reset camera")
        btn_cam.clicked.connect(self.camera.reset)
        r5.addWidget(btn_cam)
        r5.addWidget(QtWidgets.QLabel("Left-drag: orbit · wheel: zoom · right-drag: pan"))
        r5.addStretch(1)
        v5.addLayout(r5)
        row, self.pipe_srgb, self.pipe_acescg = self._viewport_pair()
        v5.addLayout(row, 1)
        outer.addWidget(c5, 1)
        return page

    def _wire(self) -> None:
        self.chk_matrix.toggled.connect(self.schedule_texture)
        self.chk_matrix.toggled.connect(self._sync_enabled)

        # Pipeline-tab dropdowns <-> Viewer-tab controls
        self._link_combos(self.cmb_input_cs, self.cmb_interp)
        self._link_combos(self.cmb_target, self.cmb_conv)
        self._link_combo_check(self.cmb_mat, self.chk_matrix, checked_index=1)
        self._link_combo_check(self.cmb_shmat, self.chk_shader_matrix, checked_index=1)
        self.chk_shader_matrix.toggled.connect(self.on_shader_matrix_toggled)
        for cb in (self.cmb_input_cs, self.cmb_interp):
            cb.currentIndexChanged.connect(lambda _=0: (self.schedule_texture(), self.request_previews()))
        for cb in (self.cmb_target, self.cmb_conv):
            cb.currentIndexChanged.connect(lambda _=0: self.on_target_changed())
        self._link_combos(self.cmb_view, self.cmb_view2)
        self._link_combos(self.cmb_shape, self.cmb_shape2)

        for cb in (self.cmb_view, self.cmb_view2):
            cb.currentIndexChanged.connect(lambda _=0: (self.request_redraw(), self.request_previews()))
        for cb in (self.cmb_shape, self.cmb_shape2):
            cb.currentIndexChanged.connect(lambda _=0: (self.request_redraw(), self.request_previews()))
        for cb in self.show_combos.values():
            cb.currentIndexChanged.connect(lambda _=0: self.request_previews())
        self.cmb_preview.currentIndexChanged.connect(lambda _=0: self.request_previews())
        self.cmb_snapcam.currentIndexChanged.connect(lambda _=0: self.request_previews())
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.camera.changed.connect(lambda: self.request_redraw(fast=True))
        self.settings.changed.connect(self.on_settings_changed)
        self.chk_shader.toggled.connect(self.on_shader_toggled)
        self.chk_matrix.toggled.connect(lambda _=False: self.request_redraw())
        for sb in self.spins:
            sb.valueChanged.connect(lambda _=0: self.shader_active() and self.request_redraw())

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

    # ------------------------------------------ custom view-transform shader --
    def shader_active(self) -> bool:
        return self.chk_shader.isChecked() and self.shader_ok

    def active_matrix(self) -> np.ndarray:
        """The matrix handed to the shader: the manual grid when "Use manual shader override" is ticked
        or the custom matrix is in use for the texture, else the OCIO default."""
        use_manual = self.chk_shader_matrix.isChecked() or \
            (self.target_cs() is not None and self.chk_matrix.isChecked())
        return self.matrix() if use_manual else self.ocio_matrix

    def on_shader_matrix_toggled(self, checked: bool) -> None:
        if checked and not self.chk_shader.isChecked():
            self.chk_shader.setChecked(True)      # a matrix for a disabled shader would do nothing
        self._sync_enabled()
        self.request_redraw()
        self.request_previews()

    def to709(self, ws: str) -> np.ndarray:
        if ws not in self._to709_cache:
            try:
                m = np.eye(3) if ws == self.ctx.lin709 else self.ctx.linear_matrix(ws, self.ctx.lin709)
            except Exception:  # noqa: BLE001
                m = np.eye(3)
            self._to709_cache[ws] = m
        return self._to709_cache[ws]

    def compile_shader(self) -> None:
        src = self.shader_edit.toPlainText()
        ok, log = self.gl.compile_post(src)
        self.shader_ok = ok
        if ok:
            state = "used for all previews" if self.chk_shader.isChecked() else "tick the checkbox to use it"
            self.lbl_shader.setStyleSheet("color:#4a4; font-size:11px;")
            self.lbl_shader.setText(f"Compiled OK - {state}." + (f"\n{log.strip()}" if log.strip() else ""))
            if src != self.d.view_shader:
                self.settings.set("view_shader", src)
        else:
            self.lbl_shader.setStyleSheet("color:#d55; font-size:11px;")
            self.lbl_shader.setText("Compile failed - the OCIO view transform is used instead.\n" + log.strip()[-1800:])
        self.request_redraw()
        self.request_previews()

    def on_shader_toggled(self, checked: bool) -> None:
        self.shader_panel.setVisible(checked)
        if checked and not self.shader_ok:
            self.compile_shader()
        else:
            self.request_redraw()
            self.request_previews()

    def reset_shader(self) -> None:
        self.shader_edit.setPlainText(DEFAULT_VIEW_SHADER)
        self.compile_shader()

    def render_view(self, shape: str, cam: Camera, ws: str, scene: np.ndarray, w: int, h: int,
                    texture: int | None = None, aspect: float | None = None) -> np.ndarray:
        """Render + view transform -> display-referred float RGB (OCIO view, or the custom shader)."""
        if self.shader_active():
            return self.gl.render(shape, cam, scene, w, h, texture=texture, aspect=aspect, post=True,
                                  input_matrix=self.active_matrix(), to709=self.to709(ws))
        lin = self.gl.render(shape, cam, scene, w, h, texture=texture, aspect=aspect)
        return self.ctx.to_display(np.maximum(lin, 0.0), ws, self.cmb_view.currentText())

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
        where = QtWidgets.QLabel(f"Settings file: {self.settings.path}")
        where.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        where.setStyleSheet("color:#888;")
        outer.addWidget(where)

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
        if name == "view_shader":
            return
        if name == "*":  # full reset: refresh every widget
            for n, setter in self.setting_setters.items():
                setter(getattr(d, n))
            self.shader_edit.setPlainText(d.view_shader)
            self.shader_ok = False
            if self.chk_shader.isChecked():
                self.compile_shader()
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
        if self.settings.save_error:
            self.statusBar().showMessage(self.settings.save_error, 10000)
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
        if self.target_cs() is None:        # the matrix only takes effect when converting
            self.cmb_target.setCurrentText(self.ctx.acescg)
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
        self.request_previews()

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
        self.box_render.setTitle(f"5 · Final renderers - live 3D  ({src} → primitive → view transform)")

    def on_tab_changed(self, _i: int) -> None:
        self.request_redraw()
        self.request_previews()

    # ------------------------------------------------- context / config --
    def bind_context(self, ctx: OcioContext) -> None:
        self.ctx = ctx
        self._to709_cache.clear()
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
        name = ctx.config.getName() or ctx.config.getDescription()[:60]
        self.lbl_cfg.setText(f"Config: {name}\n{ctx.source}" if ctx.source else f"Config: {name}")
        self._fill_cs_combos()
        self.update_captions()
        self.reset_matrix()
        self._sync_enabled()
        self.schedule_texture()
        self.request_previews()

    def load_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "OCIO config", str(app_dir()),
                                                        "OCIO config (*.ocio *.ocioz)")
        if not path:
            return
        try:
            ctx = OcioContext(OCIO.Config.CreateFromFile(path))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Config error", str(e))
            return
        ref = portable_ref(path)
        ctx.source = ref
        self.bind_context(ctx)
        answer = QtWidgets.QMessageBox.question(
            self, "OCIO config", f"Use this config at start-up?\n{ref}",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.settings.set("default_config", ref)
            if "default_config" in self.setting_setters:
                self.setting_setters["default_config"](ref)

    # ------------------------------------------------ colour-space combos --
    def target_cs(self) -> str | None:
        t = self.cmb_target.currentText()
        return None if (not t or t == NO_CONVERSION) else t

    def input_cs(self) -> str:
        return self.cmb_input_cs.currentText() or self.ctx.srgb_tex

    def _fill_cs_combos(self) -> None:
        ctx = self.ctx
        prev_in, prev_tg = self.cmb_input_cs.currentText(), self.cmb_target.currentText()
        for cb in (self.cmb_input_cs, self.cmb_interp):
            cb.set_config(ctx, (), keep=prev_in, default=ctx.srgb_tex)
        for cb in (self.cmb_target, self.cmb_conv):
            cb.set_config(ctx, (NO_CONVERSION,), keep=prev_tg, default=ctx.acescg)

    def on_target_changed(self) -> None:
        self._sync_enabled()
        self.ocio_matrix = self._compute_ocio_matrix()
        if not self.chk_matrix.isChecked():   # keep a custom matrix, but track the OCIO default otherwise
            self._fill_spins(self.ocio_matrix)
        self.schedule_texture()
        self.request_previews()

    def _compute_ocio_matrix(self) -> np.ndarray:
        try:
            return self.ctx.gamut_matrix(self.target_cs() or self.ctx.acescg)
        except Exception:  # noqa: BLE001
            return np.eye(3)

    def _fill_spins(self, m: np.ndarray) -> None:
        for sb, val in zip(self.spins, np.asarray(m).flatten()):
            sb.blockSignals(True)
            sb.setValue(float(val))
            sb.blockSignals(False)

    # ------------------------------------------------------------ matrix --
    def reset_matrix(self) -> None:
        self.ocio_matrix = self._compute_ocio_matrix()
        self._fill_spins(self.ocio_matrix)
        self.schedule_texture()
        self.request_previews()

    def matrix(self) -> np.ndarray:
        return np.array([sb.value() for sb in self.spins], dtype=np.float32).reshape(3, 3)

    def _sync_enabled(self) -> None:
        conv = self.target_cs() is not None
        self.chk_matrix.setEnabled(conv)
        self.cmb_mat.setEnabled(conv)
        on = (conv and self.chk_matrix.isChecked()) or self.chk_shader_matrix.isChecked()
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
        self.cmb_input_cs.setCurrentText(self.ctx.lin709 if is_float else self.ctx.srgb_tex)  # EXR/HDR are normally linear
        prev = np.clip(arr, 0, 1)
        if is_float:
            prev = prev ** (1 / 2.2)
        self.slot.set_preview(QtGui.QPixmap.fromImage(to_qimage((prev * 255 + 0.5).astype(np.uint8))))
        self.schedule_texture()
        self.request_previews()

    # -------------------------------------------- pipeline: final texture --
    def schedule_texture(self) -> None:
        self.tex_timer.start()

    def _params(self) -> PipelineParams:
        target = self.target_cs()
        return PipelineParams(self.input_cs(), target, target is not None and self.chk_matrix.isChecked(),
                              self.matrix())

    def describe_pipeline(self) -> list[str]:
        ctx, p = self.ctx, self._params()
        h, w = self.image.shape[:2]  # type: ignore[union-attr]
        log = [f"Input  : {Path(self.image_path).name}  {w}x{h}  read as '{p.input_cs}'"]
        if p.target_cs is None:
            log.append("Convert OFF -> texture = pixel values as loaded (no OCIO applied)")
        elif p.override:
            log.append(f"Decode : OCIO '{p.input_cs}' -> '{ctx.lin709}'")
            log.append(f"Matrix : custom, Linear Rec.709 -> '{p.target_cs}'\n         " + "\n         ".join(
                " ".join(f"{v: .5f}" for v in r) for r in p.matrix))
        else:
            log.append(f"Convert: OCIO '{p.input_cs}' -> '{p.target_cs}'")
        return log

    def rebuild_texture(self) -> None:
        if self.image is None:
            return
        try:
            st = run_pipeline(self.ctx, self.image, self._params())
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
                disp = self.render_view(shape, self.camera, ws, scene, w, h)
                vp.set_image(to_qimage((np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)))
        except Exception as e:  # noqa: BLE001
            self.log.setPlainText("\n".join(self.pipeline_log + [f"RENDER ERROR: {e}"]))

    def update_log(self) -> None:
        if not self.pipeline_log or self.image is None:
            return
        shown = f"step '{self.STAGE_TITLES[self.override_key]}' (override)" if self.override_key else "final texture"
        vt = ("custom HLSL shader (inputMatrix: " + ("manual" if self.chk_shader_matrix.isChecked() or
                                                    (self.target_cs() is not None and self.chk_matrix.isChecked())
                                                    else "OCIO default") + ")") if self.shader_active() else \
            f"display '{self.ctx.display}' / view '{self.cmb_view.currentText()}'"
        extra = [f"Renderers show: {shown}",
                 f"Shape  : {self.cmb_shape.currentText()}   (GPU: {self.gl.info})",
                 f"View   : working space -> {vt}",
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
        p = self._params()
        notes = {k: "" for k in self.show_combos}
        cam = self.camera if self.cmb_snapcam.currentIndex() == 1 else self.thumb_cam
        vt = "the custom HLSL shader" if self.shader_active() else f"view '{view}'"
        try:
            st = run_pipeline(ctx, self.thumb, p)
            managed = self.cmb_preview.currentIndex() == 1
            stages = {"in": (st.inp, st.inp_cs, self.th_in), "ws": (st.ws, st.ws_cs, self.th_ws),
                      "mx": (st.mx, st.mx_cs, self.th_mx), "view": (st.mx, st.mx_cs, self.th_view)}
            for key, (arr, cs, label) in stages.items():
                arr = np.nan_to_num(arr)
                mode = self.show_combos[key].currentIndex()
                if mode == 0:  # flat texture
                    if key == "view" and self.shader_active():
                        disp = self.gl.apply_view_shader(arr, self.active_matrix(), self.to709(cs))
                    elif managed or key == "view":
                        disp = ctx.to_display(np.maximum(arr, 0), cs, view)
                    else:
                        disp = arr
                else:  # render this stage's texture on the primitive with the chosen renderer
                    ws, scene = (ctx.lin709, self.scene_709) if mode == 1 else (ctx.acescg, self.scene_acescg)
                    tid = self.gl.set_stage_texture(key, arr)
                    tw, th = self.d.thumb_render_width, self.d.thumb_render_height
                    disp = self.render_view(shape, cam, ws, scene, tw, th, texture=tid,
                                            aspect=arr.shape[1] / arr.shape[0])
                    notes[key] = (f"\nRendered on {shape} by the {self.SHOW_MODES[mode]}: numbers read as "
                                  f"'{ws}', then {vt}.")
                self._set_thumb(label, disp)
        except Exception as e:  # noqa: BLE001
            self.info_view.setText(f"Preview error: {e}")
            return

        h, w = self.image.shape[:2]
        self.info_in.setText(f"{Path(self.image_path).name}\n{w}x{h} · read as '{st.inp_cs}'\n"
                             f"{fmt_rgb(st.inp)}{notes['in']}")
        if p.target_cs is not None:
            self.info_ws.setText(f"OCIO: '{p.input_cs}' → '{p.target_cs}'\n{fmt_rgb(st.ws)}{notes['ws']}")
            m, tag = (p.matrix, f"custom, Lin Rec.709 → '{p.target_cs}'") if p.override \
                else (self.ocio_matrix, "OCIO transform (linear approx.)")
            rows = "\n".join(" ".join(f"{v: .4f}" for v in r) for r in m)
            self.info_mx.setText(f"{tag}\n{rows}\n{fmt_rgb(st.mx)}{notes['mx']}")
        else:
            self.info_ws.setText(f"No conversion: values kept as loaded and read as '{ctx.lin709}'.\n"
                                 f"{fmt_rgb(st.ws)}{notes['ws']}")
            self.info_mx.setText(f"n/a (conversion is off){notes['mx']}")
        self.info_view.setText(f"'{st.mx_cs}' → " + ("custom HLSL shader" if self.shader_active()
                               else f"display '{ctx.display}', view '{view}'") + f".{notes['view']}")


def load_ocio_context(settings: Settings) -> tuple[OcioContext, list[str]]:
    """Load the start-up config: $OCIO, then the configured default, then the built-in config.
    If the configured default fails it is reset to the built-in config. Returns (context, warnings)."""
    candidates: list[tuple[str, str]] = []
    if os.environ.get("OCIO"):
        candidates.append(("$OCIO", os.environ["OCIO"]))
    configured = settings.d.default_config.strip() or BUILTIN_CONFIG
    candidates.append(("settings", configured))
    if configured != BUILTIN_CONFIG:
        candidates.append(("built-in", BUILTIN_CONFIG))

    warnings: list[str] = []
    settings_failed = False
    for label, ref in candidates:
        try:
            ctx = OcioContext(OCIO.Config.CreateFromFile(resolve_config_ref(ref)))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Could not load the {label} OCIO config '{ref}':\n{e}")
            settings_failed = settings_failed or label == "settings"
            continue
        ctx.source = ref
        if settings_failed:
            settings.set("default_config", BUILTIN_CONFIG)
            warnings.append(f"The default config in settings was reset to the built-in '{BUILTIN_CONFIG}'.")
        return ctx, warnings
    raise RuntimeError("\n\n".join(warnings))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    settings = Settings()
    try:
        ctx, warnings = load_ocio_context(settings)
    except Exception as e:  # noqa: BLE001
        QtWidgets.QMessageBox.critical(None, "OCIO error", f"Could not initialise any OCIO config:\n{e}")
        return 1
    try:
        renderer = GLRenderer(settings.d)
    except Exception as e:  # noqa: BLE001
        QtWidgets.QMessageBox.critical(None, "OpenGL error", f"Could not initialise OpenGL 3.3:\n{e}")
        return 1
    win = MainWindow(ctx, renderer, settings)
    win.show()
    if warnings:
        QtWidgets.QMessageBox.warning(win, "OCIO config", "\n\n".join(warnings))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())