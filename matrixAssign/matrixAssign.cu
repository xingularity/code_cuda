/**
Author: Zong-han, Xie <icbm0926@gmail.com>
This code is to implement a very simple CUDA example.
This example demonstrate basic assigment operation.

*/
// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <kernel.cu>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Macro to handle error
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void runTest(int argc, char** argv);

int main(int argc, char** argv){
    runTest(argc, argv);
    cudaDeviceReset();
}

void runTest(int argc, char** argv){
    cudaSetDevice(0);

    unsigned int num_blocks = 4; // number of blocks in the grid
    unsigned int num_threads = 4; // number of threads in each block

    unsigned int mem_size = sizeof(double) * num_threads * num_blocks;

    double* h_idata = (double*) malloc(mem_size);
    double* h_odata = (double*) malloc(mem_size);

    double* d_idata = (double*) malloc(mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, mem_size));
    double* d_odata = (double*) malloc(mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, mem_size));

    for (unsigned int i = 0; i < num_threads*num_blocks; ++i)
        h_idata[i] = 1.0;

    CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    dim3 grid(num_blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    testKernel<<<grid, threads, mem_size>>>(d_idata, d_odata);

    CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < num_blocks; ++i){
        for (unsigned int j = 0; j < num_threads; ++j){
            printf("%15.5lf ", h_odata[i*num_threads + j]);
        }
        printf("\n");
    }

    free(h_idata);
    free(h_odata);
    CUDA_SAFE_CALL(cudaFree(d_idata));
    CUDA_SAFE_CALL(cudaFree(d_odata));

}
