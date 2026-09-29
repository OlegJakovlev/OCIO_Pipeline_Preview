# OCIO Pipeline Viewer
A small Python desktop application for exploring image colour pipelines with OpenColorIO (OCIO).

## Requirements
- Python 3.12

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
 - Viewer   - compact controls on the left, two linked 3D viewports on the right.
 ![alt text](images/viewer_tab.png)
 
 - Pipeline - every step as a thumbnail of the texture (Input -> Working space ->
              Input matrix -> View transform), each with its own dropdown, followed by
              the two renderers. Dropdowns are synced with the Viewer-tab controls.
 ![alt text](images/pipeline_tab.png)
 
### Viewport controls:
 - left-drag = orbit
 - wheel = zoom
 - right/middle-drag = pan,
 - double-click = reset camera

### Steps to use
 1. Load image (drag & drop / click)   -> interpret as sRGB-encoded or linear/raw
 2. Optional conversion to ACEScg      -> decode to linear Rec.709, then gamut
                                          Rec.709 -> ACEScg (OCIO, or your own 3x3 matrix)
 3. View transform (OCIO display/view) -> applied to both 3D viewports
 4. The same "final texture" on a selectable 3D primitive, shown in two real-time
    OpenGL viewports: an sRGB renderer (Linear Rec.709 working space) and an
    ACEScg renderer. The cameras are linked.
	
