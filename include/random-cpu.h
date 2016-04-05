#ifndef XMATRIX_RANDOM_H_
#define XMATRIX_RANDOM_H_

#include "common.h"
#include "tensor.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <malloc.h>

namespace xmatrix {

template<>
struct Random<cpu> {
	static const bool _isCPU = cpu::_isCPU;
	static const bool _isGPU = cpu::_isGPU;

	gsl_rng *r;

	XMATRIX_INLINE Random(unsigned long seed XMATRIX_DEFAULT_SEED) {
		r = gsl_rng_alloc(gsl_rng_default);
		gsl_rng_set(r, seed);
	}

	XMATRIX_INLINE ~Random() {
		gsl_rng_free(r);
	}

	template<size_t dimension, typename DType>
	XMATRIX_INLINE void UniformInit(Tensor<cpu, dimension, DType> &t, Shape<dimension> shape, DType max = 1, DType min = 0) {
		size_t length = shape.getSize();
		t.AllocMem(shape);

		for (size_t i=0; i<length; i++)
			t._ptr[i] = gsl_rng_uniform(r) * (max - min) + min;
	}

	template<size_t dimension, typename DType>
	XMATRIX_INLINE void GaussianInit(Tensor<cpu, dimension, DType> &t, Shape<dimension> shape, DType mean = 0, DType sigma = 1) {
		size_t length = shape.getSize();
		t.AllocMem(shape);

		for (size_t i=0; i<length; i++)
			t._ptr[i] = gsl_ran_gaussian(r, sigma) + mean;
	}
};

}// namespace xmatrix

#endif // XMATRIX_RANDOM_H_