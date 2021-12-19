#include <math.h>
#include <cstdlib>
#include <cuda_runtime.h>

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void matrixMul(int ny, int nx, int nny, int nnx, float* norData, float* result) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;
    
    float v[8][8]{};
    __shared__ float xx[4][64];
    __shared__ float yy[4][64];

    for (int ks = 0; ks < nx; ks += 4) {
        int ija = ja * 8 + ia;
        int i = ic * 64 + ija;
        int j = jc * 64 + ija;
        for (int f = 0; f < 4; ++f) {
            int k = ks + f;
            xx[f][ija] = norData[i*nnx + k];
            yy[f][ija] = norData[j*nnx + k];
        }

        __syncthreads();
        
        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            float y[8];
            for (int jb = 0; jb < 8; ++jb) {
                y[jb] = yy[f][jb*8 + ja];
            }

            for (int ib = 0; ib < 8; ++ib) {
                float x = xx[f][ib*8 + ia];
                for (int jb = 0; jb < 8; ++jb) {
                    v[ib][jb] += x * y[jb];
                }
            }
        }

        __syncthreads();
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic*64 + ib*8 + ia;
            int j = jc*64 + jb*8 + ja;
            if (i < ny && j <= i) {
                result[j*ny + i] = v[ib][jb];
            }
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    int nny = roundup(ny, 64);
    int nnx = roundup(nx, 4);
    float* norData = new float[nny*nnx]{0};
    
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
            norData[row*nnx + col] = normal;
            sqsum += normal * normal;
        }

        float sumsqrt = sqrt(sqsum);
        for (int col = 0; col < nx; col++) {
            norData[row*nnx + col] = norData[row*nnx + col] / sumsqrt;
        }
    }

    float* norDataGPU = NULL;
    cudaMalloc((void**)&norDataGPU, nny * nnx * sizeof(float));
    cudaMemcpy(norDataGPU, norData, nny * nnx * sizeof(float), cudaMemcpyHostToDevice);
    float* resultGPU = NULL;
    cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float));
    cudaMemcpy(resultGPU, result, ny * ny * sizeof(float), cudaMemcpyHostToDevice);

    // matrix multiplication
    dim3 dimBlock(8, 8);
    dim3 dimGrid(nny / 64, nny / 64);
    matrixMul<<<dimGrid, dimBlock>>>(ny, nx, nny, nnx, norDataGPU, resultGPU);

    // finish
    cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(norDataGPU);
    cudaFree(resultGPU);
    delete[] norData;
}
