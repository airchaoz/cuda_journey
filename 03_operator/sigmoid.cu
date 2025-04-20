#include "bits/stdc++.h"

using namespace std;

#define WARP_SIZE 32
#define FLOAT4(x) (reinterpret_cast<float4 *> (&(x))[0])
#define MAX_EXP_F32 88.37
#define MIN_EXP_F32 -88.37
#define MAX_EXP_F16 __float2half(11.08)
#define MIN_EXP_F16 __float2half(-9.7)


__global__ void sigmod_f32(float *x, float *y, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        float val = x[tid];
        val = fmin(fmax(val, MAX_EXP_F32), MIN_EXP_F32);
        y[tid] = 1.0f / (1.0f + expf(-val));
    }
}

__global__ void sigmod_f32x4(float *x, float *y, int N) {

    int tid = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    float4 reg_a = FLOAT4(x[tid]);
    float4 reg_b;

    reg_a.x = fmin(fmax(reg_a.x, MAX_EXP_F32), MIN_EXP_F32);
    reg_a.y = fmin(fmax(reg_a.y, MAX_EXP_F32), MIN_EXP_F32);
    reg_a.z = fmin(fmax(reg_a.z, MAX_EXP_F32), MIN_EXP_F32);
    reg_a.w = fmin(fmax(reg_a.w, MAX_EXP_F32), MIN_EXP_F32);

    reg_b.x = 1.0f / (1.0f + expf(-reg_a.x));
    reg_b.y = 1.0f / (1.0f + expf(-reg_a.y));
    reg_b.z = 1.0f / (1.0f + expf(-reg_a.z));
    reg_b.w = 1.0f / (1.0f + expf(-reg_a.w));

    if (tid < N) { FLOAT4(y[tid]) = reg_b; }
}