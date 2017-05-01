#include <iostream>

// Required to include CUDA vector 
#include <cuda_runtime.h>

class CUDADeviceProperties{
public:
    CUDADeviceProperties(int device);
    void printCudaDevice();

private:
    cudaDeviceProp prop;
};
