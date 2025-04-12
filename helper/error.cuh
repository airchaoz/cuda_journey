#ifndef ERROR_CUH
#define ERROR_CUH
#include <stdio.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA ERROR:\n");                                                 \
      printf("    File:%s\n", __FILE__);                                       \
      printf("    Line:%d\n", __LINE__);                                       \
      printf("    Error code:%d\n", error_code);                               \
      printf("    Error msg:%s\n", cudaGetErrorString(error_code));            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)                                                                  \

#endif
