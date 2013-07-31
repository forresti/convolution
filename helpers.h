#ifndef __HELPERS_H__
#define __HELPERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0) 

double read_timer();
cv::Mat getRawImage(string in_filename);
void outputProcessedImage(unsigned int* processedImg, int width, int height, string out_filename);
void outputProcessedImageFloat(float* processedImg, int width, int height, string out_filename);
void outputProcessedImageUchar(uchar* processedImg, int width, int height, string out_filename);
float* getDummyImg(int height, int width);
uchar* getDummyImgUchar(int height, int width);
float getResponseTime(cudaEvent_t start, cudaEvent_t stop);

#endif

