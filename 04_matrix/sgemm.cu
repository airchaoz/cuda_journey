#include "bits/stdc++.h"
#include "cublas_v2.h"

using namespace std;

#define OFFSET(row, col, N) ((row) * (N) + (col))

void sgemm_cpu(float *a, float *b, float *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.f;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__ void sgemm_v1(float *a, float *b, float *c, int M, int N, int K) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int m = tid / N;
    int n = tid % N;

    if (m < M && n < N) {
        float r_sum = 0.f;
        for (int k = 0; k < K; k++) {
            r_sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = r_sum;
    }
}


void fill_random(float *arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

int main() {

    int M = 1024, N = 1024, K = 1024;
    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    float *c = (float*)malloc(M * N * sizeof(float));

    fill_random(h_a, M * K);
    fill_random(h_b, K * N);

    sgemm_cpu(h_a, h_b, c, M, N, K);

    float *d_a, *d_b, *d_c;
    float *h_c = (float*)malloc(M * N * sizeof(float));
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * K *sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (M * N + blockSize - 1) / blockSize;
    sgemm_v1<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = -FLT_MAX;
    for (int i = 0; i < M * N; i++) {
        float this_error = fabs(c[i] - h_c[i]);
        max_err = fmax(max_err, this_error);
    }
    printf("max error is %f\n", max_err);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(c);
    free(h_c);
}