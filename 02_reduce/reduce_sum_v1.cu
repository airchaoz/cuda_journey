#include "bits/stdc++.h"
#include "../helper/error.cuh"

using namespace std;

/**
  *@brief: This function computes the sum of an array using OpenMP.
  */
void reduce_sum_omp(float *arr, float *result, int n) {

    float sum = 0.f;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    *result = sum;
}

/**
  *@brief: Using shared memory to compute the sum of an array.
  */
__global__ void reduce_sum_v1(float *arr, float *result, int n) {

    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    s_data[tid] = (idx < n) ? arr[idx] : 0.f;
    __syncthreads();

    for (int i = blockDim.x >> 1; i > 0; i = i >> 1) {
        if (tid < i) {
            s_data[tid] += s_data[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = s_data[0];
    }
}

void test_error() {

    const int n = 100'000;
    const int block_size = 128;
    const int grid_size = (n + block_size - 1) / block_size;

    float *h_a = (float *)malloc(n * sizeof(float));
    float *h_result = (float *)malloc(grid_size * sizeof(float));
    float *omp_result = (float *)malloc(sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / float(RAND_MAX);
    }
    reduce_sum_omp(h_a, omp_result, n);

    float *d_a;
    float *d_result;

    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_result, grid_size * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v1<<<grid_size, block_size, block_size * sizeof(float)>>>(d_a, d_result, n);

    cudaMemcpy(h_result, d_result, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float v1_result = 0.f;
    for (int i = 0; i < grid_size; i++) {
        v1_result += h_result[i];
    }

    printf("the result of omp is %f\n", *omp_result);
    printf("the result of v1  is %f\n", v1_result);

    free(h_a);
    free(h_result);
    free(omp_result);

    cudaFree(d_result);
    cudaFree(d_a);
}

int main() {
    test_error();
}