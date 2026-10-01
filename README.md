# OCIO Pipeline Viewer
A small Python desktop application for exploring image colour pipelines with OpenColorIO (OCIO).

## License
DO WHAT THE HELL YOU WANT - Code is not owned by me, and was produced by utilizing AI agents.

## Requirements
- Python 3.11 / 3.12

Install the required packages:
```bash
py -3.12 -m pip install PySide6 PyOpenGL opencolorio numpy pillow OpenEXR opencv-python
```

## Run
```bash
py -3.12 ocio_pipeline_viewer.py
```

## Application
### Tabs
 - Viewer   - controls on the left, with possibility to modify matrix conversion and view transform using custom HLSL shader. sRGB and ACEScg 3D realtime viewports on the right.
 ![alt text](images/viewer_tab.png)
 
 - Pipeline - every conversion step with a thumbnail, with possibility to choose renderer.
 ![alt text](images/pipeline_tab.png)
 
### Viewport controls:
 - left-drag = orbit
 - wheel = zoom
 - right/middle-drag = pan,
 - double-click = reset camera

### Steps to use
 1. Load image (drag & drop / click)
 2. Select workspace defined in your OCIO config, or optionally override via custom input matrix or HLSL shader
 3. Select View transform (OCIO display/view) -> applied to both 3D viewports, or use custom HLSL shader override
 4. Select 3D primitive
 5. Observe Linear Rec.709 and an ACEScg viewports
