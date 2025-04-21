#include "bits/stdc++.h"

using namespace std;

#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

__global__ void relu_f32(float *x, float *y, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        y[tid] = fmaxf(0.0f, x[tid]);
    }
}

__global__ void relu_f32x4(float *x, float *y, int N) {
    int tid = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (tid < N) {
        float4 reg_a = FLOAT4(x[tid]);

        reg_a.x = fmaxf(reg_a.x, 0.0f);
        reg_a.y = fmaxf(reg_a.y, 0.0f);
        reg_a.z = fmaxf(reg_a.z, 0.0f);
        reg_a.w = fmaxf(reg_a.w, 0.0f);

        FLOAT4(y[tid]) = reg_a;
    }
}