# OpticalEncoder


**OpticalEncoder** is a Python project based on Pytorch framework that allows to place various optical elements (lenses, slits, apertures, etc.) and perform wave optics propagation to simulate phenomena such as diffraction and compute PSF (Point Spread Function) with GPU.

## Directory Structure
```bash
OpticalEncoder/
├─ opticalencoder/
│   ├─ __init__.py
│   ├─ propagation.py      # Wave propagation methods
│   ├─ elements.py         # Definitions for optical elements (lens, slit, etc.)
│   └─ utils.py            # Miscellaneous utility functions
├─ examples/
│   ├─ diffraction_example.py
│   └─ lens_example.py
├─ requirements.txt
└─ README.md
```


## TODO List

[ ] Verificaiton of utilization of GPU devices, and Parallelization  
[ ] Scalable ASM (SAS)  
[ ] Various phase initialization methods.  
[ ] Considering shifted locations of sources or fields.  
[ ] Considering various directions of the input fields.  
[ ] More various examples.  

## Examples
1. Spatially filtered PSF using the pinhole.
