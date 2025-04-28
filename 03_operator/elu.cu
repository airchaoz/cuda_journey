#include "bits/stdc++.h"

using namespace std;


__global__ void elu_f32(float *x, float *y, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        y[tid] = x[tid] > 0.f ? x[tid] : expf(x[tid]) - 1.0f;
    }
}