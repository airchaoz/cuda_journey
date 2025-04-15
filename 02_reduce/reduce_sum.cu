#include "bits/stdc++.h"

using namespace std;

/**
  *@brief: This function computes the sum of an array
  */
  float kahan_sum_single(const float* arr, int n) {
    float sum = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < n; i++) {
        float y = arr[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
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
        atomicAdd(result, s_data[0]);
    }
}

__global__ void reduce_sum_v2(float *arr, float *result, int n) {

    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    s_data[tid] = (idx < n) ? arr[idx] : 0.f;
    __syncthreads();

    for (int i = blockDim.x >> 1; i >= 32; i = i >> 1) {
        if (tid < i) {
            s_data[tid] += s_data[tid + i];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = s_data[tid];

        for (int mask = 16; mask > 0; mask >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, mask);
        }

        if (tid == 0) {
            atomicAdd(result, val);
        }
    }
}

void test_error(int n) {

    const int block_size = 128;
    const int grid_size = (n + block_size - 1) / block_size;

    float *h_a = (float *)malloc(n * sizeof(float));
    float *v1_result = (float *)malloc(sizeof(float));
    float *v2_result = (float *)malloc(sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / float(RAND_MAX);
    }
    float kahan_result = kahan_sum_single(h_a, n);

    float *d_a;
    float *d_v1_result;
    float *d_v2_result;

    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_v1_result, sizeof(float));
    cudaMalloc((void **)&d_v2_result, sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v1<<<grid_size, block_size, block_size * sizeof(float)>>>(d_a, d_v1_result, n);
    cudaMemcpy(v1_result, d_v1_result, sizeof(float), cudaMemcpyDeviceToHost);

    reduce_sum_v2<<<grid_size, block_size, block_size * sizeof(float)>>>(d_a, d_v2_result, n);
    cudaMemcpy(v2_result, d_v2_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("the result of kahan is %f\n", kahan_result);
    printf("the result of v1    is %f\n", *v1_result);
    printf("the result of v2    is %f\n", *v2_result);

    free(h_a);
    free(v1_result);
    free(v2_result);

    cudaFree(d_v1_result);
    cudaFree(d_v2_result);
    cudaFree(d_a);
}

float test_performance(int n, void (*gpu_reduce_sum) (float *arr, float *result, int n)) {

    const int block_size = 128;
    const int grid_size = (n + block_size - 1) / block_size;

    float *h_a = (float *)malloc(n * sizeof(float));
    float *h_result = (float *)malloc(grid_size * sizeof(float));

    srand(42);
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / float(RAND_MAX);
    }

    float *d_a;
    float *d_result;

    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_result, grid_size * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int test_rounds = 5;
    for (int i = 0; i < test_rounds; i++) {
        gpu_reduce_sum<<<grid_size, block_size, block_size * sizeof(float)>>>(d_a, d_result, n);
    }

    cudaMemcpy(h_result, d_result, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("The vector reduce_sum_v1(size %d) time elapsed: %f ms\n", n, milliseconds / float(test_rounds));

    free(h_a);
    free(h_result);

    cudaFree(d_result);
    cudaFree(d_a);
    return milliseconds / float(test_rounds);
}

int main() {
    const int n = 100'000'000;

    test_error(n);
    float milliseconds;
    milliseconds = test_performance(n, reduce_sum_v1);
    printf("The vector reduce_sum_v1(size %d) time elapsed: %f ms\n", n, milliseconds);
    milliseconds = test_performance(n, reduce_sum_v2);
    printf("The vector reduce_sum_v2(size %d) time elapsed: %f ms\n", n, milliseconds);
}