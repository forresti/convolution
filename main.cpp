#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <set>
#include <map>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda.h"
#include <opencv2/opencv.hpp>

#include "convRunner.h"
#include "helpers.h"

using namespace std;
using namespace cv;

void deviceStuff();

int main (int argc,char **argv)
{
    deviceStuff();

    //testConvolution_withDummyImg(640, 480);
    //testConvolution_withDummyImg(512, 512);
    testConvolution_withDummyImg(9000, 9000);
    //testConvolution();

    return 0;
}

//hard-code device preferences here.
void deviceStuff()
{
    cudaSetDevice(2);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("deviceName = %s \n", prop.name);
}

