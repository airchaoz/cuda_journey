#include "bits/stdc++.h"

using namespace std;

__global__ void vadd(float *a, float *b, float *c, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  int nums_n[] = {128, 512, 1024, 2048, 4096};
  int n_size = sizeof(nums_n) / sizeof(int);

  for (int i = 0; i < n_size; i++) {
    int n = nums_n[i];

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_a, n * sizeof(float));

    vadd<<<(n + 256) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
}
