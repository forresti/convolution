echo '#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "convolution.h"
//#include "helpers.h"
using namespace std;
//using namespace cv;

texture<float, 2, cudaReadModeElementType> tex;

static inline __device__ int clamp(int x, int l, int r)
{
    //max and min are gpu intrinsics
    return max(l, min(x,r));
}

static inline __device__ int clamp_addr(int x, int y, int mx, int my)
{
    int xaddr = clamp(x, 0, (mx-1));
    int yaddr = clamp(y, 0, (my-1));
    return (yaddr*mx) + xaddr;
}'

#load2x2 up to load7x7.
# user inputs the startX and startY.
# using the standard anchor, startX = globalX - kernelSize/2
echo "//Load from DRAM to register"
for((amountToLoad=2; amountToLoad<8; amountToLoad++)) do
echo "__device__ void load_global_register_size${amountToLoad}x${amountToLoad}(const float* in, float registers[$amountToLoad][$amountToLoad], int startX, int startY, int width, int height)
{
    for(int i=0; i<$amountToLoad; i++)
    {"
        for((j=0; j<$amountToLoad; j++)) do
            echo "        registers[i][$j] = in[clamp_addr(startX+$j, startY+i, width, height)];"
        done
echo "    }
}"
done

for((amountToLoad=2; amountToLoad<8; amountToLoad++)) do
echo "__device__ void load_texCache_register_size${amountToLoad}x${amountToLoad}(const float* in, float registers[$amountToLoad][$amountToLoad], int startX, int startY, int width, int height)
{
    for(int i=0; i<$amountToLoad; i++)
    {"
        for((j=0; j<$amountToLoad; j++)) do
            echo "        registers[i][$j] =  tex2D(tex, float(startX+$j)+0.5, float(startY+i)+0.5);"
        done
echo "    }
}"
done

echo "//Store from register to DRAM"
for((amountToLoad=1; amountToLoad<8; amountToLoad++)) do
echo "
__device__ void store_register_global_size${amountToLoad}x${amountToLoad}(float* out, float registers[$amountToLoad][$amountToLoad], int startX, int startY, int width, int height)
{
    for(int i=0; i<$amountToLoad; i++)
    {"
        for((j=0; j<$amountToLoad; j++)) do
                echo "        out[(startY+i)*width + startX+$j] = registers[i][$j];"
            done
echo "    }
}"
done

filters[2]=.2500; filters[3]=0.1111; filters[4]=0.0625; filters[5]=0.0400; filters[6]=0.0278; filters[7]=0.0204; #1/(kernelSize*kernelSize)
echo "//Convolution -- hard-coded 2D arrays"
for((inDim=2; inDim<8; inDim++)) do
for((kernelSize=2; kernelSize<8; kernelSize++)) do #convolution filter dim
if [ $inDim -ge $kernelSize ]
then
echo "
//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size${inDim}x${inDim}_kernel${kernelSize}x${kernelSize}(float in[$inDim][$inDim], float *out, int startX, int startY)
{
    const float filter = ${filters[${kernelSize}]}; // 1/(kernelSize 2D)
    float tmp=" 
for((y=0; y<$kernelSize; y++)) do
    for((x=0; x<$kernelSize; x++)) do
        echo "        in[startY+$y][startX+$x] * filter +"
    done
    echo ""
done
    echo "        0;
    *out = tmp;
}"
fi
done
done

#notes:
# const int convsPerThread = $amountToLoad - (ghost zone)
# ghost zone = convSize/2 (possibly +1)
echo "//Kernels that load, convolve, and store."
for memoryScheme in "global_register" "texCache_register"
do
for((kernelSize=2; kernelSize<=7; kernelSize++)) do
for((amountToLoad=2; amountToLoad<=7; amountToLoad++)) do
sqrtConvsPerThread=$(($amountToLoad-$kernelSize+1)) #square of this is total number of convs per thread [TODO: check correctness] 
if [ $sqrtConvsPerThread -ge 1 ]
then
echo "__global__ void convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(const float* in, float* out, const int width, const int height)
{
    int globalX = $sqrtConvsPerThread*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = $sqrtConvsPerThread*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[$amountToLoad][$amountToLoad];
    float outRegisters[$sqrtConvsPerThread][$sqrtConvsPerThread];
    if(globalX < width && globalY < height)
    {
        load_${memoryScheme}_size${amountToLoad}x${amountToLoad}(in, registers, globalX-$((($kernelSize+1)/2-1)), globalY-$((($kernelSize+1)/2-1)), width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<$sqrtConvsPerThread; y++) //(int, y) = top left of region to convolve 
        {"
        for((x=0; x<$sqrtConvsPerThread; x++)) do
echo "           convolutionDevice_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(registers, &outRegisters[y][$x], $x, y);"
        done
echo "        }
        store_register_global_size${sqrtConvsPerThread}x${sqrtConvsPerThread}(out, outRegisters, globalX, globalY, width, height);
    }
}"
fi
done
done
done

#Benchmark the time used to do the *actual compute work*
# (not for "production" use)
echo "//Kernels that load, convolve, and store."
for memoryScheme in "global_register" "texCache_register"
do
for((kernelSize=2; kernelSize<=7; kernelSize++)) do
amountToLoad=$kernelSize
sqrtConvsPerThread=1 
if [ $sqrtConvsPerThread -ge 1 ]
then
echo "__global__ void convolutionKernel_${memoryScheme}_BenchmarkComputeTime_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = $sqrtConvsPerThread*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = $sqrtConvsPerThread*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[$amountToLoad][$amountToLoad];
    float outRegisters[$sqrtConvsPerThread][$sqrtConvsPerThread];
    if(globalX < width && globalY < height)
    {
        load_${memoryScheme}_size${amountToLoad}x${amountToLoad}(in, registers, globalX-$((($kernelSize+1)/2-1)), globalY-$((($kernelSize+1)/2-1)), width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<$sqrtConvsPerThread; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {"
        for((x=0; x<$sqrtConvsPerThread; x++)) do
echo "           convolutionDevice_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(registers, &outRegisters[y][$x], $x, y);"
        done
echo "        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size${sqrtConvsPerThread}x${sqrtConvsPerThread}(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}"
fi
done
done

#global only kernel, hard-coded bounds
memoryScheme="global_only"
for((kernelSize=2; kernelSize<8; kernelSize++)) do
startOffset=$((($kernelSize+1)/2-1))
for((sqrtConvsPerThread=1; sqrtConvsPerThread<8; sqrtConvsPerThread++)) do
amountToLoad=$(($sqrtConvsPerThread+$kernelSize-1))
echo "
//@param height,width = dims of 'in' array
__global__ void convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(const float* in, float* out, const int width, const int height)
{
    int globalX = $sqrtConvsPerThread*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = $sqrtConvsPerThread*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = ${filters[${kernelSize}]}; // 1/(kernelSize 2D)
        for(int yy=0; yy < $sqrtConvsPerThread; yy++)
        {"
            #for(int outIdxX=globalX; outIdxX < globalX+$sqrtConvsPerThread; outIdxX++) //TODO: unroll this in script.
            for((xx=0; xx<$sqrtConvsPerThread; xx++)) do
echo "            
                outIdxX = globalX+$xx;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=" 
for((y=-$startOffset; y<$(($kernelSize-$startOffset)); y++)) do
    for((x=-$startOffset; x<$(($kernelSize-$startOffset)); x++)) do
        echo "                    in[clamp_addr(outIdxX+$x, outIdxY+$y, width, height)] * filter +" #TODO: use currX and currY
    done
    echo ""
done
    echo "                    0;"
done
echo " 
        }
    }
}"
done
done

#texCache only kernel, hard-coded bounds
memoryScheme="texCache_only"
for((kernelSize=2; kernelSize<8; kernelSize++)) do
amountToLoad=$kernelSize
echo "
//@param height,width = dims of 'in' array
__global__ void convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - $((($kernelSize+1)/2-1));
    int startY = globalY - $((($kernelSize+1)/2-1));

    if(globalX < width && globalY < height)
    {
        const float filter = ${filters[${kernelSize}]}; // 1/(kernelSize 2D)
        float tmp=" 
for((y=0; y<$kernelSize; y++)) do
    for((x=0; x<$kernelSize; x++)) do
        echo "            tex2D(tex, float(startX+$x)+0.5, float(startY+$y)+0.5) * filter +"
    done
    echo ""
done
    echo "            0;
        out[globalY*width + globalX] = tmp;
    }
}"
done

echo "//configuration directly pulled from simpleTexture in nvidia sdk
void setTexCacheParams()
{
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
}"

echo "float convolutionWrapper(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename)
{
    dim3 grid;
    dim3 block;
    block.x = 16;
    block.y = 16;
    int sqrtConvsPerThread = amountToLoad-kernelSize+1;
    int nx = width / (block.x*sqrtConvsPerThread); //magic number is for persistant kernels (e.g. if persistant6x6, divide by 4)
    int ny = height / (block.y*sqrtConvsPerThread);
    grid.x = (width % block.x == 0) ? nx : nx+1;
    grid.y = (height % block.y == 0) ? ny : ny+1;

    float *dImg;
    CHECK_CUDART(cudaMalloc((void**)&dImg, sizeof(float)*width*height)); //for input data (possibly use texture cache?)
    CHECK_CUDART(cudaMemcpy(dImg, hImg, sizeof(float)*width*height, cudaMemcpyHostToDevice));
    float* dResult; //device memory for output
    CHECK_CUDART(cudaMalloc((void**)&dResult, sizeof(float)*width*height));

    float* dFpuTime;
    CHECK_CUDART(cudaMalloc((void**)&dFpuTime, sizeof(float)*width*height));
    CHECK_CUDART(cudaMemset(dFpuTime, 0, sizeof(float)*width*height));

    double start = read_timer();"

    memoryScheme="global_only" 
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((sqrtConvsPerThread=1; sqrtConvsPerThread<8; sqrtConvsPerThread++)) do
    amountToLoad=$(($sqrtConvsPerThread+$kernelSize-1))
echo "    if(amountToLoad == $amountToLoad && kernelSize == $kernelSize && memoryScheme == \"${memoryScheme}\") {
        cudaFuncSetCacheConfig(convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}, cudaFuncCachePreferL1);
        convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}<<<grid, block>>>(dImg, dResult, width, height);
    }"
    done
    done
    memoryScheme="global_register_BenchmarkComputeTime" 
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
        amountToLoad=$kernelSize
echo "    if(amountToLoad == $amountToLoad && kernelSize == $kernelSize && memoryScheme == \"${memoryScheme}\") {
        cudaFuncSetCacheConfig(convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}, cudaFuncCachePreferL1);
        convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }"
    done

    memoryScheme="global_register" #TODO: if I do global_shared_register, I should probably set 'prefer shared' 
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((amountToLoad=$kernelSize; amountToLoad<8; amountToLoad++)) do
echo "    if(amountToLoad == $amountToLoad && kernelSize == $kernelSize && memoryScheme == \"${memoryScheme}\") {
        cudaFuncSetCacheConfig(convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}, cudaFuncCachePreferL1);
        convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}<<<grid, block>>>(dImg, dResult, width, height);
    }"
#echo "    if(amountToLoad == $amountToLoad && memoryScheme == \"global_shared_register\"){ 
#        cudaFuncSetCacheConfig(convolutionKernel_global_shared_register_persist${amountToLoad}x${amountToLoad}_kernel3x3, cudaFuncCachePreferShared);
#        convolutionKernel_global_shared_register_persist${amountToLoad}x${amountToLoad}_kernel3x3 <<< grid, block >>>(dImg, dResult, width, height);
#    }"
    done
    done
echo '    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
    
    CHECK_CUDART(cudaFree(dImg));
    if(outputImgFlag)
    {
        float* hResult = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hResult, dResult, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        outputProcessedImageFloat(hResult, width, height, outFilename);
        free(hResult);
    }
    if(memoryScheme.find("BenchmarkComputeTime") != string::npos)
    {
        float* hFpuTime = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hFpuTime, dFpuTime, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        printf("COMPUTE time on GPU, %dx%d filter: \n", kernelSize, kernelSize);
        for(int i=0; i<(width*height); i++)
        {
            printf("%f\n", hFpuTime[i]);
        }
    }
    CHECK_CUDART(cudaFree(dResult));
    CHECK_CUDART(cudaFree(dFpuTime));
    return responseTime;
}'

echo "float convolutionWrapper_texCache(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename)
{
    dim3 grid;
    dim3 block;
    block.x = 16;
    block.y = 16;
    int sqrtConvsPerThread = amountToLoad-kernelSize+1;
    int nx = width / (block.x*sqrtConvsPerThread); //magic number is for persistant kernels (e.g. if persistant6x6, divide by 4)
    int ny = height / (block.y*sqrtConvsPerThread);
    grid.x = (width % block.x == 0) ? nx : nx+1;
    grid.y = (height % block.y == 0) ? ny : ny+1;

    cudaArray *dImg; //cudaArray*, not float*
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CHECK_CUDART(cudaMallocArray(&dImg, &channelDesc, width, height));
    CHECK_CUDART(cudaMemcpyToArray(dImg, 0, 0, hImg, width*height*sizeof(float), cudaMemcpyHostToDevice));
    setTexCacheParams();
    CHECK_CUDART(cudaBindTextureToArray(tex, dImg, channelDesc));
    float* dummy = NULL; //my texCache code doesn't need input data ptr, but it was easier to have one in code generation anyway.
    float* dResult; //device memory for output
    CHECK_CUDART(cudaMalloc((void**)&dResult, sizeof(float)*width*height));

    double start = read_timer();"

    memoryScheme="texCache_register" #TODO if I do texCache_shared_register, I should probably set 'prefer shared' 
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((amountToLoad=$kernelSize; amountToLoad<8; amountToLoad++)) do
echo "    if(amountToLoad == $amountToLoad && kernelSize == $kernelSize && memoryScheme == \"${memoryScheme}\") {
        cudaFuncSetCacheConfig(convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}, cudaFuncCachePreferL1);
        convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}<<<grid, block>>>(dummy, dResult, width, height);
    }"
    done
    done
    memoryScheme="texCache_only" #TODO if I do texCache_shared_register, I should probably set 'prefer shared' 
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    amountToLoad=$kernelSize
echo "    if(amountToLoad == $amountToLoad && kernelSize == $kernelSize && memoryScheme == \"${memoryScheme}\") {
        cudaFuncSetCacheConfig(convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}, cudaFuncCachePreferL1);
        convolutionKernel_${memoryScheme}_size${amountToLoad}x${amountToLoad}_kernel${kernelSize}x${kernelSize}<<<grid, block>>>(dummy, dResult, width, height);
    }"
    done
echo '
    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
   
    CHECK_CUDART(cudaFreeArray(dImg));
    if(outputImgFlag)
    {
        float* hResult = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hResult, dResult, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        outputProcessedImageFloat(hResult, width, height, outFilename);
        free(hResult);
    }
    CHECK_CUDART(cudaFree(dResult));
    return responseTime;
}'


