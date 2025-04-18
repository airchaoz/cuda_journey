#include "bits/stdc++.h"
#include "../helper/error.cuh"

using namespace std;

#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])

__global__ void vadd(float *a, float *b, float *c, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

__global__ void vadd_f32x4(float *a, float *b, float *c, int N) {
  int tid = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
  if (tid < N) {
    float4 reg_a = FLOAT4(a[tid]);
    float4 reg_b = FLOAT4(a[tid]);
    float4 reg_c = FLOAT4(a[tid]);

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    FLOAT4(c[tid]) = reg_c;
  }
}

void test_performance(int n) {

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  float *h_a = (float *)malloc(n * sizeof(float));
  float *h_b = (float *)malloc(n * sizeof(float));
  float *h_c = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / static_cast<float>(RAND_MAX);
    h_a[i] = rand() / static_cast<float>(RAND_MAX);
  }

  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_c, n * sizeof(float)));

  CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int test_rounds = 5;
  for (int i = 0; i < test_rounds; i++) {
    vadd<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
  }

  CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("The vector add(size %d) time elapsed: %f ms\n", n, milliseconds / float(test_rounds));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}

void test_max_err(int n) {

  float *h_a = (float *)malloc(n * sizeof(float));
  float *h_b = (float *)malloc(n * sizeof(float));
  float *h_c = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    h_a[i] = rand() / static_cast<float>(RAND_MAX);
    h_a[i] = rand() / static_cast<float>(RAND_MAX);
  }

  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_c, n * sizeof(float)));

  CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

  int test_rounds = 5;
  float max_err = 0;
  for (int i = 0; i < test_rounds; i++) {
    vadd<<<(n + 256) / 256, 256>>>(d_a, d_b, d_c, n);
    CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; j++) {
      max_err = max(max_err, abs(h_c[j] - (h_a[j] + h_b[j])));
    }
  }

  printf("The vector add(size %d) max err: %f\n", n, max_err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}

int main() {
  int nums_n[] = {128, 512, 1024, 2048, 4096};
  int n_size = sizeof(nums_n) / sizeof(int);

  for (int i = 0; i < n_size; i++) {
    test_performance(nums_n[i]);
    test_max_err(nums_n[i]);
  }
}
