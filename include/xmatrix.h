#ifndef XMATRIX_H_
#define XMATRIX_H_

/**
* Default Compile Setups
*/
#ifndef XMATRIX_USE_CUDA
#define XMATRIX_USE_CUDA 1
#ifndef XMATRIX_USE_CUDNN
#define XMATRIX_USE_CUDNN 1
#endif
#endif
#ifndef XMATRIX_USE_MKL
#define XMATRIX_USE_MKL 1
#endif

#include "common.h"
#include "tensor.h"
#include "tensor-wrapper.h"

#if XMATRIX_USE_MKL == 0
#include "tensor-cpu.h"
#else
#include "tensor-mkl.h"
#endif

#if XMATRIX_USE_CUDA == 1
#include "tensor-cuda.h"
#endif

/**
* Include all xmatrix header files
*/

#endif // XMATRIX_COMMON_H_