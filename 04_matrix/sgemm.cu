#include "bits/stdc++.h"
#include "cublas_v2.h"

using namespace std;

#define OFFSET(row, col, N) ((row) * (N) + (col))
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])

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

__global__ void sgemm_v2(float *a, float *b, float *c, int M, int N, int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    float r_c[TM][TN] = { 0.f };

    int load_a_smem_m = tid / 2;
    int load_a_smem_k = (tid % 2) * 4;
    int load_b_smem_k = tid / 32;
    int load_b_smem_n = (tid % 32) * 4;

    int load_a_gmem_m = load_a_smem_m + blockIdx.y * BM;
    int load_b_gmem_n = load_b_smem_n + blockIdx.x * BN;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = load_a_smem_k + bk * BK;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = load_b_smem_k + bk * BK;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += s_a[threadIdx.y * TM + m][k] * s_b[k][threadIdx.x * TN + n];
                }
            }
        }

        __syncthreads();

        for (int m = 0; m < TM; m++) {
            int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + m;
            for (int n = 0; n < TN; n++) {
                int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + n;
                c[OFFSET(store_c_gmem_m, store_c_gmem_n, N)] = r_c[m][n];
            }
        }
    }
}


void fill_random(float *arr, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

float* get_cpu_result(int M, int N, int K) {

    float *a = (float*)malloc(M * K * sizeof(float));
    float *b = (float*)malloc(K * N * sizeof(float));
    float *c = (float*)malloc(M * N * sizeof(float));

    fill_random(a, M * K);
    fill_random(b, K * N);

    sgemm_cpu(a, b, c, M, N, K);

    free(a);
    free(b);
    return c;
}

float* get_v1_result(int M, int N, int K){

    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    float *h_c = (float*)malloc(M * N * sizeof(float));

    fill_random(h_a, M * K);
    fill_random(h_b, K * N);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * K *sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (M * N + blockSize - 1) / blockSize;
    sgemm_v1<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);

    return h_c;
}

float* get_v2_result(int M, int N, int K) {

    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    float *h_c = (float*)malloc(M * N * sizeof(float));

    fill_random(h_a, M * K);
    fill_random(h_b, K * N);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * K *sizeof(float), cudaMemcpyHostToDevice);

    int BN = 128, BM = 128;
    int TN = 8, TM = 8;
    dim3 blockSize(BN / TN, BM / TM);
    dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_v2<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);

    return h_c;
}

float get_max_err(float *a, float *b, int N) {
    float max_err = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        float this_error = fabs(a[i] - b[i]);
        max_err = fmax(max_err, this_error);
    }
    return max_err;
}

int main() {

    int M = 1024, N = 1024, K = 1024;
    float* cpu_c = get_cpu_result(M, N, K);
    float* v1_c = get_v1_result(M, N, K);
    float* v2_c = get_v2_result(M, N, K);

    float v1_err = get_max_err(cpu_c, v1_c, M * N);
    float v2_err = get_max_err(cpu_c, v2_c, M * N);
    printf("v1 max error is %f\n", v1_err);
    printf("v2 max error is %f\n", v2_err);

    free(cpu_c);
    free(v1_c);
    free(v2_c);
}