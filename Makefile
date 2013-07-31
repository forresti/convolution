#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = main.o convolution.o convRunner.o helpers.o

EXENAME = main

CC = nvcc 
CCOPTS = `pkg-config opencv --cflags`  -c  -g -pg -O0 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -Xptxas -v #-gencode=arch=compute_35,code=sm_35 -Xptxas -v  
LINK = nvcc
LINKOPTS = `pkg-config opencv --libs` -g -pg -o

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) $(LINKOPTS) $(EXENAME) $(OBJS)

main.o : main.cpp convRunner.h helpers.h
	$(CC) $(CCOPTS) main.cpp

convRunner.o : convRunner.cu convRunner.h convolution.h
	$(CC) $(CCOPTS) convRunner.cu

convolution.o : convolution.cu convolution.h
	$(CC) $(CCOPTS) convolution.cu

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp


clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

