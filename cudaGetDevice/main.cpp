#include <iostream>
#include "CUDADeviceProperties.hpp"

int main(int argc, char **argv)
{
    CUDADeviceProperties cuda_device_info(0);
    cuda_device_info.printCudaDevice();
    return 0;
}

