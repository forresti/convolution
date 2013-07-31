<h5>This fast convolution implementation accompanies the following paper</h5>

Forrest N. Iandola, David Sheffield, Michael Anderson, Phitchaya Mangpo Phothilimthana, and Kurt Keutzer. 
"Communication-Minimizing 2D Convolution in GPU Registers." 
To appear in IEEE International Conference on Image Processing (ICIP), 2013. 


<h3>Recommended System Configuration</h3>
- Ubuntu 12.04
- NVIDIA Fermi or Kepler GPU (compute capability 2.0 or higher)
- CUDA 5.0
- OpenCV installed for file I/O


<h3>Getting started</h3>
``` C++
make
./main
```

<h3>Code structure</h3>

``` C++
__global__ convolutionKernel_filter3x3_size4x4(...)
    load 4x4 window to registers
    
    compute 4 convolution windows, 3x3 each.
    
    save results to global memory
```
