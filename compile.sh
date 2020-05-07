#!/bin/bash

module load cuda

SRCDIR=/home_nfs/murilo1/gabriel/cap4/openmp+cuda/


cd $SRCDIR

nvcc -Xcompiler -fopenmp  -gencode arch=compute_20,code=sm_20 $1 -o $2 -I$CUDA_INC -L$CUDA_LIB -lcudart --std=c++11
