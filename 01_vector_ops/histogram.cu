#include "bits/stdc++.h"

using namespace std;

#define WARP_SIZE 32
#define INT4(val) (reinterpret_cast<int4*> (&(val))[0])

__global__ void histogram(int *a, int *r, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&r[a[tid]], 1);
    }
}

/**
 * @note the length of input vec must be a multiple of 4
 */
__global__ void histogram_i32x4(int *a, int *r, int N) {
    int tid = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < N) {
        int4 reg_a = INT4(a[tid]);
        atomicAdd(&r[reg_a.x], 1);
        atomicAdd(&r[reg_a.y], 1);
        atomicAdd(&r[reg_a.z], 1);
        atomicAdd(&r[reg_a.w], 1);
    }
}