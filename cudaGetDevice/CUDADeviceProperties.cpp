#include "CUDADeviceProperties.hpp"

CUDADeviceProperties:: CUDADeviceProperties(int device){
    cudaGetDeviceProperties(&prop, 0);
}

void CUDADeviceProperties::printCudaDevice(){
    std::cout << "GPU name: " << prop.name << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem/1024/1024 << "MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock/1024.0 << "KB" << std::endl;
    std::cout << "Registry per Block: " << prop.regsPerBlock/1024.0 << "KB" << std::endl;
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

    std::cout << "Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: " << prop.canMapHostMemory << std::endl;
    std::cout << "Compute mode: " << prop.computeMode << std::endl;
    std::cout << "https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__TYPES_g7eb25f5413a962faad0956d92bae10d0.html#g7eb25f5413a962faad0956d92bae10d0" << std::endl;
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
    std::cout << "Device Overlap: " << prop.deviceOverlap << std::endl;
    std::cout << "If device has ECC support enabled: " << prop.ECCEnabled << std::endl;
    std::cout << "Device is integrated as opposed to discrete: " << prop.integrated << std::endl;
    std::cout << "Kernel Execute Timeout Enabled: " << prop.kernelExecTimeoutEnabled << std::endl;
    std::cout << "Max Texture 1D: " << prop.maxTexture1D << std::endl;
    std::cout << "Maximum 2D texture dimensions: ";
    for (int i = 0; i < 2; ++i){
        std::cout << prop.maxTexture2D[i] << " ";
    }
    std::cout << std::endl;
    /*
    std::cout << "Maximum 2D texture array dimensions: ";
    for (int i = 0; i < 3; ++i){
        std::cout << prop.maxTexture2DArray[i] << " ";
    }
    std::cout << std::endl;
    */
    std::cout << "Maximum 3D texture dimensions: ";
    for (int i = 0; i < 3; ++i){
        std::cout << prop.maxTexture3D[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Number of multiprocessors on device: " << prop.multiProcessorCount << std::endl;
    std::cout << "PCIBusID: " << prop.pciBusID << std::endl;
    std::cout << "PCIDeviceID: " << prop.pciDeviceID << std::endl;
    std::cout << "Alignment requirements for surfaces: " << prop.surfaceAlignment << std::endl;
    std::cout << "TCC Driver, 1 if device is a Tesla device using TCC driver, 0 otherwise: " << prop.tccDriver << std::endl;

}
