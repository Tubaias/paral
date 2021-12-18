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

__global__ void matrixMul2(int ny, int nx, float* norData, float* result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j > i) { return; }

    for (int k = 0; k < nx; k++) {
        float x = norData[i*nx + k];
        float y = norData[j*nx + k];
        result[j*ny + i] += x * y;
    }
}

__global__ void matrixMul(int ny, int nx, int nny, float* norData, float* result) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    for (int k = 0; k < nx; k++) {
        float x[8];
        float y[8];

        for (int ib = 0; ib < 8; ++ib) {
            int i = ic*64 + ib*8 + ia;
            x[ib] = norData[i*nx + k];
        }

        for (int jb = 0; jb < 8; ++jb) {
            int j = jc*64 + jb*8 + ja;
            y[jb] = norData[j*nx + k];
        }

        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                int i = ic*64 + ib*8 + ia;
                int j = jc*64 + jb*8 + ja;
                if (i < ny && j <= i) {
                    result[j*ny + i] += x[ib] * y[jb];
                }
            }
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    int nny = roundup(ny, 64);
    float* norData = new float[nny*nx]{0};
    
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
            norData[row*nx + col] = norData[row*nx + col] / sumsqrt;
        }
    }

    float* norDataGPU = NULL;
    CHECK(cudaMalloc((void**)&norDataGPU, nny * nx * sizeof(float)));
    CHECK(cudaMemcpy(norDataGPU, norData, nny * nx * sizeof(float), cudaMemcpyHostToDevice));
    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(resultGPU, result, ny * ny * sizeof(float), cudaMemcpyHostToDevice));

    // matrix multiplication
    dim3 dimBlock(8, 8);
    dim3 dimGrid(nny / 64, nny / 64);
    matrixMul<<<dimGrid, dimBlock>>>(ny, nx, nny, norDataGPU, resultGPU);
    CHECK(cudaGetLastError());

    // finish
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(norDataGPU));
    CHECK(cudaFree(resultGPU));
    delete[] norData;
}
