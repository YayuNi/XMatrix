#ifndef XMATRIX_COMMON_H_
#define XMATRIX_COMMON_H_

/**
* Default inline macro
*/
#ifdef _MSC_VER
#define XMATRIX_INLINE __forceinline
#else
#define XMATRIX_INLINE inline __attribute__((always_inline))
#endif

/**
* Datatype and constant definition
*/
#ifndef NULL
#define NULL 0
#endif

#define XMATRIX_DEFAULT_SEED = gsl_rng_default_seed

/**
* include system header files
*/
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdarg>
#include <typeinfo>
#include <malloc.h>
#include <math.h>

using namespace std;

#endif // XMATRIX_COMMON_H_