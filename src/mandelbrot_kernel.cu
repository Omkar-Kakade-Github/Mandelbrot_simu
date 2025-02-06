#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <surface_functions.h>
#include <stdio.h>  // For printf
#include <string.h> // For memset

extern "C" {

__global__ void mandelbrotKernelSurface(cudaSurfaceObject_t surfaceOutput,
                                        int width, int height,
                                        float minRe, float maxRe,
                                        float minIm, float maxIm,
                                        int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;

    // Mandelbrot calculation (unchanged)
    float c_re = minRe + (maxRe - minRe) * x / width;
    float c_im = minIm + (maxIm - minIm) * y / height;

    float z_re = 0.0f, z_im = 0.0f;
    int iterations = 0;
    while (z_re * z_re + z_im * z_im <= 4.0f && iterations < maxIterations) {
        float new_re = z_re * z_re - z_im * z_im + c_re;
        float new_im = 2.0f * z_re * z_im + c_im;
        z_re = new_re;
        z_im = new_im;
        iterations++;
    }

    uchar4 color;
    if (iterations == maxIterations) {
        color = make_uchar4(0, 0, 0, 255);
    } else {
        unsigned char val = static_cast<unsigned char>(255 * iterations / maxIterations);
        color = make_uchar4(val, 0, 255 - val, 255);
    }
    
    surf2Dwrite(color, surfaceOutput, x * sizeof(uchar4), (height - 1 - y));
}

void launchMandelbrotKernelSurface(cudaArray* textureArray, int width, int height,
                                   float minRe, float maxRe,
                                   float minIm, float maxIm,
                                   int maxIterations) {
    // Create surface object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObj;
    cudaError_t err = cudaCreateSurfaceObject(&surfaceObj, &resDesc);
    if (err != cudaSuccess) {
        printf("cudaCreateSurfaceObject failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Pass surface object as first parameter
    mandelbrotKernelSurface<<<gridSize, blockSize>>>(
        surfaceObj,  // Surface object as first argument
        width, height,
        minRe, maxRe,
        minIm, maxIm,
        maxIterations
    );

    cudaDeviceSynchronize();
    cudaDestroySurfaceObject(surfaceObj);
}

} // extern "C"
