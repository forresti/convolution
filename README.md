Recommended System Configuration:
- Ubuntu 12.04
- NVIDIA Fermi or Kepler GPU
- OpenCV installed for file I/O


Getting started:
``` C++
make
./main
```

Code structure:

``` C++
__global__ convolutionKernel_filter3x3_size4x4(...)
    load 4x4 window to registers
    
    compute 4 convolution windows, 3x3 each.
    
    save results to global memory
```
