#ifndef XMATRIX_MODEL_H_
#define XMATRIX_MODEL_H_

#include "tensor.h"

namespace xmatrix {

template<typename device>
struct LogsticRegression {
	static_assert(is_base_of<AbstractDevice, device>::value, "Target device not supported!");

	Vector<cpu, double>::type w;
	Scalar<cpu, double>::type b;

	Matrix<cpu, double>::type x;
	Vector<cpu, int>::type y;

	Vector<cpu, int>::type &_predict;
	Vector<cpu, double>::type &_cost;

	double _lambda;

	XMATRIX_INLINE LogsticRegression(double lambda) 
		: _predict(Vector<cpu, int>::null), _cost(Vector<cpu, double>::null), _lambda(lambda) {

		Vector<cpu, double>::type &p_1 = 1 / (1 + op::Exp(-x*w - b));
		_predict = p_1 > 0.5;

		Vector<cpu, double>::type &cross_entropy = -y * op::Log(p_1) - (1-y) * op::Log(1 - p_1);
		_cost = op::Mean(cross_entropy) + _lambda * op::Sum(op::Pow(w, 2));
	}

	// Model Train
	XMATRIX_INLINE void Train(double *features, Shape<2> features_shape, int *labels, Shape<1> label_shape) {
	}

	// Model Predict
	XMATRIX_INLINE int Predict(double *features) {
		return 0;
	}

};

} // namespace xmatrix

#endif