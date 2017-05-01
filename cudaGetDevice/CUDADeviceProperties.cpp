#include "CUDADeviceProperties.hpp"

CUDADeviceProperties:: CUDADeviceProperties(int device){
    cudaGetDeviceProperties(&prop, 0);
}

void CUDADeviceProperties::printCudaDevice(){
    std::cout << "GPU name: " << prop.name << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem/1024.0/1024.0 << "MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock/1024.0/1024.0 << "MB" << std::endl;
    std::cout << "Registry per Block: " << prop.regsPerBlock/1024.0/1024.0 << "MB" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Pitch: " << prop.memPitch/1024.0/1024 << "MB" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension: ";
    for (int i = 0; i < 3; ++i){
        std::cout << prop.maxThreadsDim[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Max Grid Size: ";
    for (int i = 0; i < 3; ++i){
        std::cout << prop.maxGridSize[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << std::endl;
    std::cout << "Total Const Memory: " << prop.totalConstMem << "B" << std::endl;
    std::cout << "Computation Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Texture Alignment: " << prop.textureAlignment << "B" << std::endl;

}


