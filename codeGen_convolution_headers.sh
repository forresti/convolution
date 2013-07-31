
echo '#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;

float convolutionWrapper(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename="convolved.png");
float convolutionWrapper_texCache(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename="convolved.png");
'

for memoryScheme in "global_register" "texCache_register"  
do
    for((kernelSize=2; kernelSize<=7; kernelSize++)) do
        for((amountToLoad=2; amountToLoad<=7; amountToLoad++)) do
            sqrtConvsPerThread=$(($amountToLoad-$kernelSize+1)) #square of this is total number of convs per thread [TODO: check correctness] 
            if [ $sqrtConvsPerThread -ge 1 ]
            then
                echo "__global__ void convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(const float* in, float* out, const int width, const int height);"
            fi
        done
    done
    echo ""
done
echo "__global__ void convolutionKernel_global_only_kernel3x3(const float* in, float* out, const int width, const int height);"
echo "#endif"


