#ifndef XMATRIX_TENSOR_GSL_H_
#define XMATRIX_TENSOR_GSL_H_

#include "common.h"
#include "tensor.h"

namespace xmatrix {

/**
* Add Operator
*/
template<size_t dimension_dest, typename DType_dest,
	size_t dimension_lhs, typename DType_lhs,
	size_t dimension_rhs, typename DType_rhs>
struct AddTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE AddTensor(
		Tensor<cpu, dimension_lhs, DType_lhs> &lhs, 
		Tensor<cpu, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs>
		(lhs, rhs) { }

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] + _rhs._ptr[i];
		}
	}
};

/**
* Add Operator: Tensor = Tensor + Scalar or Scalar + Tensor
*/
template<size_t dimension, typename DType_dest, typename DType_lhs, typename DType_rhs>
struct AddTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE AddTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] + _rhs._ptr[0];
		}
	}
};

/**
* Minus Operator
*/
template<size_t dimension_dest, typename DType_dest,
	size_t dimension_lhs, typename DType_lhs,
	size_t dimension_rhs, typename DType_rhs>
struct MinusTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE MinusTensor(
		Tensor<cpu, dimension_lhs, DType_lhs> &lhs, 
		Tensor<cpu, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension_dest, DType_dest, cpu, dimension_lhs, DType_lhs, cpu, dimension_rhs, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] - _rhs._ptr[i];
		}
	}
};

/**
* Minus Operator: Tensor = Tensor - Scalar
*/
template<size_t dimension, typename DType_dest, typename DType_lhs, typename DType_rhs>
struct MinusTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE MinusTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] - _rhs._ptr[0];
		}
	}
};

/**
* Multiple Operator: Vector = Vector x Matrix
*/
template<typename DType_dest, typename DType_lhs, typename DType_rhs>
struct MultipleTensor<cpu, 1, DType_dest, cpu, 1, DType_lhs, cpu, 2, DType_rhs> 
	: public BinaryDeducedTensor<cpu, 1, DType_dest, cpu, 1, DType_lhs, cpu, 2, DType_rhs> {

	XMATRIX_INLINE MultipleTensor(
		Tensor<cpu, 1, DType_lhs> &lhs, 
		Tensor<cpu, 2, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, 1, DType_dest, cpu, 1, DType_lhs, cpu, 2, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape[0] == _rhs._shape[0]);
			AllocMem(Shape1(_rhs._shape[1]));

			for (size_t i = 0; i < _shape[0]; i++) {
				_ptr[i] = 0;
				for (size_t j = 0; j < _lhs._shape[0]; j++)
					_ptr[i] += _lhs._ptr[j] * _rhs._ptr[j * _rhs._stride + i];
			}
		}
	}
};

/**
* Multiple Operator: Matrix = Matrix x Matrix
*/
template<typename DType_dest, typename DType_lhs, typename DType_rhs>
struct MultipleTensor<cpu, 2, DType_dest, cpu, 2, DType_lhs, cpu, 2, DType_rhs> 
	: public BinaryDeducedTensor<cpu, 2, DType_dest, cpu, 2, DType_lhs, cpu, 2, DType_rhs> {

	XMATRIX_INLINE MultipleTensor(
		Tensor<cpu, 2, DType_lhs> &lhs, 
		Tensor<cpu, 2, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, 2, DType_dest, cpu, 2, DType_lhs, cpu, 2, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape[1] == _rhs._shape[0]);
			AllocMem(Shape2(_lhs._shape[0], _rhs._shape[1]));

			for (size_t i = 0; i < _shape[0]; i++) 
				for (size_t j = 0; j < _shape[1]; j++) {
					_ptr[i * _stride + j] = 0;
					for (size_t k = 0; k < _lhs._shape[1]; k++)
						_ptr[i * _stride + j] += _lhs._ptr[i * _lhs._stride + k] * _rhs._ptr[k * _rhs._stride + j];
				}
		}
	}
};

/**
* Multiple Operator: Tensor = Tensor x Scalar or Scalar x Tensor
*/
template<size_t dimension, typename DType_dest, typename DType_lhs, typename DType_rhs>
struct MultipleTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE MultipleTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] * _rhs._ptr[0];
		}
	}
};

/**
* Divide Operator: Tensor = Tensor / Scalar
*/
template<size_t dimension, typename DType_dest, typename DType_lhs, typename DType_rhs>
struct DivideTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE DivideTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[i] / _rhs._ptr[0];
		}
	}
};

/**
* Divide Operator: Tensor = Scalar / Tensor
*/
template<size_t dimension, typename DType_dest, typename DType_lhs, typename DType_rhs>
struct DivideTensor<cpu, dimension, DType_dest, cpu, 0, DType_lhs, cpu, dimension, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, 0, DType_lhs, cpu, dimension, DType_rhs> {

	XMATRIX_INLINE DivideTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, dimension, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, DType_dest, cpu, 0, DType_lhs, cpu, dimension, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			for (size_t i=0; i<_rhs._shape.getSize(); i++)
				_ptr[i] = _lhs._ptr[0] / _rhs._ptr[i];
		}
	}
};

/**
* Divide Operator: Scalar = Scalar / Scalar
*/
template<typename DType_dest, typename DType_lhs, typename DType_rhs>
struct DivideTensor<cpu, 0, DType_dest, cpu, 0, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, 0, DType_dest, cpu, 0, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE DivideTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, 0, DType_dest, cpu, 0, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			_ptr[0] = _lhs._ptr[0] / _rhs._ptr[0];
		}
	}
};

/**
* Transpose Tensor : Matrix Transpose
*/
template<typename DType>
struct TransposeTensor<cpu, 2, DType, cpu, 2, DType>
	: public UnaryDeducedTensor<cpu, 2, DType, cpu, 2, DType> {
	
	XMATRIX_INLINE TransposeTensor(Tensor<cpu, 2, DType> &src) 
		: UnaryDeducedTensor<cpu, 2, DType, cpu, 2, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(Shape2(_src._shape[1], _src._shape[0]));
			for (size_t i = 0; i < _shape[0]; i++)
				for (size_t j = 0; j < _shape[1]; j++)
					_ptr[i * _stride + j] = _src._ptr[j * _src._stride + i];
		}
	}
};

/**
* Transpose Tensor : Vector Transpose
*/
template<typename DType>
struct TransposeTensor<cpu, 2, DType, cpu, 1, DType>
	: public UnaryDeducedTensor<cpu, 2, DType, cpu, 1, DType> {
	
	XMATRIX_INLINE TransposeTensor(Tensor<cpu, 1, DType> &src) 
		: UnaryDeducedTensor<cpu, 2, DType, cpu, 1, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(Shape2(_src._shape[0]), 1);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = _src._ptr[i];
		}
	}
};

/**
* Subscript Tensor
*/
template<size_t dimension_dest, size_t dimension_src, typename DType>
struct SubscriptTensor<cpu, dimension_dest, DType, cpu, dimension_src, DType>
	: public UnaryDeducedTensor<cpu, dimension_dest, DType, cpu, dimension_src, DType> {
	
	static_assert(dimension_dest + 1 == dimension_src, "Error: dimension_dest + 1 != dimension_src");

	const size_t _index;
	
	XMATRIX_INLINE SubscriptTensor(Tensor<cpu, dimension_src, DType> &src, size_t index) 
		: UnaryDeducedTensor<cpu, dimension_dest, DType, cpu, dimension_src, DType>(src), _index(index) {}
	
	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			_shape = _src._shape.SubShape();
			_stride = _shape.SubShape().getSize();
			_ptr = _src._ptr + _index * _src._stride;
		}
	}
};

/**
* Exponential Tensor
*/
template<size_t dimension, typename DType>
struct ExponentialTensor<cpu, dimension, double, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType> {
	
	XMATRIX_INLINE ExponentialTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = exp(_src._ptr[i]);
		}
	}
};

/**
* Log Tensor
*/
template<size_t dimension, typename DType>
struct LogTensor<cpu, dimension, double, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType> {
	
	XMATRIX_INLINE LogTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = log(_src._ptr[i]);
		}
	}
};

/**
* Log10 Tensor
*/
template<size_t dimension, typename DType>
struct Log10Tensor<cpu, dimension, double, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType> {
	
	XMATRIX_INLINE Log10Tensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = log10(_src._ptr[i]);
		}
	}
};

/**
* Sqrt Tensor
*/
template<size_t dimension, typename DType>
struct SqrtTensor<cpu, dimension, double, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType> {
	
	XMATRIX_INLINE SqrtTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = sqrt(_src._ptr[i]);
		}
	}
};

/**
* Power Tensor
*/
template<size_t dimension, typename DType>
struct PowerTensor<cpu, dimension, double, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType> {

	const double _exp;
	
	XMATRIX_INLINE PowerTensor(Tensor<cpu, dimension, DType> &src, double exp) 
		: UnaryDeducedTensor<cpu, dimension, double, cpu, dimension, DType>(src), _exp(exp) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = pow(_src._ptr[i], _exp);
		}
	}
};

/**
* Abs Tensor
*/
template<size_t dimension, typename DType>
struct AbsTensor<cpu, dimension, DType, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, DType, cpu, dimension, DType> {
	
	XMATRIX_INLINE AbsTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, DType, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = fabs(_src._ptr[i]);
		}
	}
};

template<size_t dimension>
struct AbsTensor<cpu, dimension, int, cpu, dimension, int>
	: public UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, int> {
	
	XMATRIX_INLINE AbsTensor(Tensor<cpu, dimension, int> &src) 
		: UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, int>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = abs(_src._ptr[i]);
		}
	}
};

/**
* Floor Tensor
*/
template<size_t dimension, typename DType>
struct FloorTensor<cpu, dimension, int, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType> {
	
	XMATRIX_INLINE FloorTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
	#pragma warning(disable: 4244)	
					_ptr[i] = (int)floor(_src._ptr[i]);
		}
	}
};

/**
* Ceil Tensor
*/
template<size_t dimension, typename DType>
struct CeilTensor<cpu, dimension, int, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType> {
	
	XMATRIX_INLINE CeilTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
#pragma warning(disable: 4244)	
				_ptr[i] = (int)ceil(_src._ptr[i]);
		}
	}
};

/**
* Round Tensor
*/
template<size_t dimension, typename DType>
struct RoundTensor<cpu, dimension, int, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType> {
	
	XMATRIX_INLINE RoundTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
#ifdef _MSC_VER
#pragma warning(disable: 4244)	
				_ptr[i] = (int)floor(_src._ptr[i] + 0.5);
#else
#pragma warning(disable: 4244)	
				_ptr[i] = (int)round(_src._ptr[i]);
#endif
		}
	}
};

/**
* GreaterThan Operator
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct GreaterThanTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs> {

	XMATRIX_INLINE GreaterThanTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, dimension, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs>
		(lhs, rhs) { }

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] > _rhs._ptr[i])? 1 : 0;
		}
	}
};

/**
* GreaterThan Operator: Tensor = Tensor > Scalar or Scalar < Tensor
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct GreaterThanTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE GreaterThanTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] > _rhs._ptr[0]) ? 1 : 0;
		}
	}
};

/**
* GreaterThan Operator: Tensor = Tensor < Scalar or Scalar > Tensor
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct GreaterThanTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs> {

	XMATRIX_INLINE GreaterThanTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, dimension, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			for (size_t i=0; i<_rhs._shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[0] > _rhs._ptr[i]) ? 1 : 0;
		}
	}
};

/**
* GreaterThan Operator: Tensor = Scalar > Scalar
*/
template<typename DType_lhs, typename DType_rhs>
struct GreaterThanTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE GreaterThanTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			_ptr[0] = (_lhs._ptr[0] > _rhs._ptr[0]) ? 1 : 0;
		}
	}
};

/**
* Equal Operator
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct EqualTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs> {

	XMATRIX_INLINE EqualTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, dimension, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, dimension, DType_rhs>
		(lhs, rhs) { }

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] == _rhs._ptr[i])? 1 : 0;
		}
	}
};

/**
* Equal Operator: Tensor = Tensor == Scalar or Scalar == Tensor
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct EqualTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE EqualTensor(
		Tensor<cpu, dimension, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_lhs._shape);
			for (size_t i=0; i<_lhs._shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] == _rhs._ptr[0]) ? 1 : 0;
		}
	}
};

/**
* Equal Operator: Tensor = Tensor == Scalar or Scalar == Tensor
*/
template<size_t dimension, typename DType_lhs, typename DType_rhs>
struct EqualTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs>
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs> {

	XMATRIX_INLINE EqualTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, dimension, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, 0, DType_lhs, cpu, dimension, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			for (size_t i=0; i<_rhs._shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[0] == _rhs._ptr[i]) ? 1 : 0;
		}
	}
};

/**
* Equal Operator: Tensor = Scalar == Scalar
*/
template<typename DType_lhs, typename DType_rhs>
struct EqualTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs> 
	: public BinaryDeducedTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs> {

	XMATRIX_INLINE EqualTensor(
		Tensor<cpu, 0, DType_lhs> &lhs, 
		Tensor<cpu, 0, DType_rhs> &rhs)
	: BinaryDeducedTensor<cpu, 0, int, cpu, 0, DType_lhs, cpu, 0, DType_rhs>
		(lhs, rhs) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			AllocMem(_rhs._shape);
			_ptr[0] = (_lhs._ptr[0] == _rhs._ptr[0]) ? 1 : 0;
		}
	}
};

/**
* And Operator
*/
template<size_t dimension>
struct AndTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int> {

	XMATRIX_INLINE AndTensor(
		Tensor<cpu, dimension, int> &lhs, 
		Tensor<cpu, dimension, int> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int>
		(lhs, rhs) { }

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] && _rhs._ptr[i]) ? 1:0;
		}
	}
};

/**
* Or Operator
*/
template<size_t dimension>
struct OrTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int> 
	: public BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int> {

	XMATRIX_INLINE OrTensor(
		Tensor<cpu, dimension, int> &lhs, 
		Tensor<cpu, dimension, int> &rhs)
	: BinaryDeducedTensor<cpu, dimension, int, cpu, dimension, int, cpu, dimension, int>
		(lhs, rhs) { }

	XMATRIX_INLINE virtual void Update() {
		if (!_isUpdated) {
			BinaryDeducedTensor::Update();
			assert(_lhs._shape == _rhs._shape);
			AllocMem(_lhs._shape);

			for (size_t i = 0; i < _shape.getSize(); i++)
				_ptr[i] = (_lhs._ptr[i] || _rhs._ptr[i]) ? 1:0;
		}
	}
};

/**
* Not Operator
*/
template<size_t dimension, typename DType>
struct NotTensor<cpu, dimension, int, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, dimension, DType, cpu, dimension, DType> {
	
	XMATRIX_INLINE NotTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, dimension, DType, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem(_src._shape);
			for (size_t i = 0; i < _shape.getSize(); i++)
					_ptr[i] = (_src._ptr[i] > 0)? 0 : 1;
		}
	}
};

/**
* Sum Opeartor
*/
template<size_t dimension, typename DType>
struct SumTensor<cpu, 0, DType, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, 0, DType, cpu, dimension, DType> {
	
	XMATRIX_INLINE SumTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, 0, DType, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem();
			_ptr[0] = 0;
			for (size_t i = 0; i < _src._shape.getSize(); i++)
					_ptr[0] += _src._ptr[i];
		}
	}
};

/**
* Mean Opeartor
*/
template<size_t dimension, typename DType>
struct MeanTensor<cpu, 0, DType, cpu, dimension, DType>
	: public UnaryDeducedTensor<cpu, 0, DType, cpu, dimension, DType> {
	
	XMATRIX_INLINE MeanTensor(Tensor<cpu, dimension, DType> &src) 
		: UnaryDeducedTensor<cpu, 0, DType, cpu, dimension, DType>(src) {}

	XMATRIX_INLINE void virtual Update() {
		if (!_isUpdated) {
			UnaryDeducedTensor::Update();
			AllocMem();
			_ptr[0] = 0;
			for (size_t i = 0; i < _src._shape.getSize(); i++)
					_ptr[0] += _src._ptr[i];
			_ptr[0] /= _shape.getSize();
		}
	}
};

} // namespace xmatrix

#endif // XMATRIX_TENSOR_GSL_H_