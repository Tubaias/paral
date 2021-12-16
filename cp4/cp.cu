#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

__global__ void matrixMul(int ny, int nx, float* norData, float* result) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= ny || j > i) { return; }

    for (int k = 0; k < nx; k++) {
        float x = norData[i*nx + k];
        float y = norData[j*nx + k];
        result[j*ny + i] += x * y;
    }
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    float* norData = new float[ny*nx];
    
    // normalization
    for (int row = 0; row < ny; row++) {
        float sum = 0.0;
        float sqsum = 0.0;

        for (int col = 0; col < nx; col++) {
            sum += data[row*nx + col];
        }

        float mean = sum / nx;

        for (int col = 0; col < nx; col++) {
            float normal = data[row*nx + col] - mean;
            norData[row*nx + col] = normal;
            sqsum += normal * normal;
        }

        float sumsqrt = sqrt(sqsum);

        for (int col = 0; col < nx; col++) {
            float buzz = norData[row*nx + col] / sumsqrt;
            norData[row*nx + col] = buzz;
        }
    }

    float* norDataGPU = NULL;
    CHECK(cudaMalloc((void**)&norDataGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(norDataGPU, norData, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(resultGPU, result, ny * ny * sizeof(float), cudaMemcpyHostToDevice));

    // matrix multiplication
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    matrixMul<<<dimGrid, dimBlock>>>(ny, nx, norDataGPU, resultGPU);
    CHECK(cudaGetLastError());

    // results
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(norDataGPU));
    CHECK(cudaFree(resultGPU));

    delete[] norData;
}
