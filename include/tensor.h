#ifndef XMATRIX_TENSOR_H_
#define XMATRIX_TENSOR_H_

#include "common.h"

namespace xmatrix {
/**
* Device Definition
*/
struct AbstractDevice {};

struct cpu : public AbstractDevice {
	static const bool _isCPU = true;
	static const bool _isGPU = false;
};

#if XMATRIX_USE_CUDA == 1
struct gpu : public AbstractDevice {
	static const bool _isCPU = false;
	static const bool _isGPU = true;
};
#endif

/**
* Tensor Shape Definition
*/
template<size_t dimension>
struct Shape {
	static const size_t _kDim = dimension;
	size_t _shape[_kDim];

	XMATRIX_INLINE Shape() {
		for (size_t i=0; i<_kDim; i++)
			_shape[i] = 1;
	}

	XMATRIX_INLINE Shape(const Shape<dimension> &d) {
		for (size_t i=0; i<_kDim; i++)
			_shape[i] = d[i];
	}

	XMATRIX_INLINE size_t getSize() const {
		size_t s = _shape[0];
		for (size_t i=1; i<_kDim; i++)
			s *= _shape[i];
		return s;
	}

	XMATRIX_INLINE size_t &operator[](size_t index) {
		return _shape[index];
	}

	XMATRIX_INLINE const size_t &operator[](size_t index) const {
		return _shape[index];
	}

	XMATRIX_INLINE bool operator==(const Shape<dimension> &d) const {
		for (size_t i=0; i<_kDim; i++)
			if (d[i] != _shape[i]) return false;
		return true;
	}

	XMATRIX_INLINE bool operator!=(const Shape<dimension> &d) const {
		return !(*this == d);
	}

	XMATRIX_INLINE Shape<dimension - 1> SubShape() const {
		Shape<dimension - 1> *s = new Shape<dimension - 1>();
		for (size_t i=0; i<dimension-1; i++)
			s->_shape[i] = _shape[i+1];
		return *s;
	}
}; // struct Shape<dimension>

template<size_t dimension>
XMATRIX_INLINE std::ostream &operator<<(std::ostream &os, Shape<dimension> d) {
	os << "Shape" << d._kDim << "(";
	os << d[0];
	for (size_t i=1; i<d._kDim; i++)
		os << ", " << d[i];
	os << ")";
	return os;
}

template<>
struct Shape<0> {
	static const size_t _kDim = 0;
	size_t _shape[1];

	XMATRIX_INLINE size_t getSize() const {
		return 1;
	}

	XMATRIX_INLINE Shape<0> SubShape() {
		return Shape<0>();
	}

	XMATRIX_INLINE size_t operator[](size_t index) const {
		return 1;
	}

	XMATRIX_INLINE bool operator==(const Shape<0> &d) const {
		return true;
	}

	XMATRIX_INLINE bool operator!=(const Shape<0> &d) const {
		return false;
	}
}; // struct Shape<0>

XMATRIX_INLINE std::ostream &operator<<(std::ostream &os, Shape<0> d) {
	os << "Shape0()";
	return os;
}

XMATRIX_INLINE Shape<0> Shape0() {
	Shape<0> *s = new Shape<0>();
	return *s;
}

XMATRIX_INLINE Shape<1> Shape1(size_t s0) {
	Shape<1> *s = new Shape<1>();
	(*s)[0] = s0;
	return *s;
}

XMATRIX_INLINE Shape<2> Shape2(size_t s0, size_t s1) {
	Shape<2> *s = new Shape<2>();
	(*s)[0] = s0; (*s)[1] = s1;
	return *s;
}

/**
* Random Definition
*/
template<typename device>
struct Random {
	static_assert(is_base_of<AbstractDevice, device>::value, "Target device not supported!");

	static const bool _isCPU = device::_isCPU;
	static const bool _isGPU = device::_isGPU;
};

/**
* Tensor Definition
*/
template<typename device, size_t dimension, typename DType>
struct Tensor {
	static_assert(is_base_of<AbstractDevice, device>::value, "Target device not supported!");
	static_assert(is_integral<DType>::value || is_floating_point<DType>::value, "DType supports integral and float point only!");
	static const bool _isCPU = device::_isCPU;
	static const bool _isGPU = device::_isGPU;

	static const size_t _kDim = dimension;
	const bool _isLeaf;
	
	Shape<dimension> _shape;
	size_t _stride;
	DType *_ptr;

	bool _isUpdated;
	
	XMATRIX_INLINE Tensor(bool isLeaf = true) : _ptr(NULL), _isLeaf(isLeaf), _isUpdated(false) {}

	XMATRIX_INLINE virtual ~Tensor() { FreeMem(); }

	XMATRIX_INLINE void Input(DType * pData, Shape<dimension> shape) {
		Invalid();
		AllocMem(shape);
		if (_isCPU) {
			memcpy(_ptr, pData, _shape.getSize() * sizeof(DType));
		}
	}

	XMATRIX_INLINE void AllocMem(Shape<dimension> shape) {
		Invalid();
		FreeMem();
		_shape = shape;
		_stride = shape.SubShape().getSize();
		if (_isCPU)
			_ptr = (DType*)calloc(_shape.getSize(), sizeof(DType));
	}

	XMATRIX_INLINE void FreeMem() {
		if (_ptr != NULL) {
			if (_isCPU)
				free(_ptr);
		}
		_ptr = NULL;
	}

	XMATRIX_INLINE Tensor<device, dimension - 1, DType> &operator[](size_t index) const {
		Tensor<device, dimension - 1, DType> *t = new SubscriptTensor<device, dimension - 1, DType, device, dimension, DType>(*this, index);
		return *t;
	}

	XMATRIX_INLINE Tensor<device, dimension - 1, DType> &operator[](size_t index) {
		Tensor<device, dimension - 1, DType> *t = new SubscriptTensor<device, dimension - 1, DType, device, dimension, DType>(*this, index);
		return *t;
	}

	XMATRIX_INLINE virtual void Update() {
		_isUpdated = true;
	}

	XMATRIX_INLINE virtual void Invalid() {
		_isUpdated = false;
	}
}; // struct Tensor

template<typename device, typename DType>
struct Tensor<device, 0, DType> {
	static_assert(is_base_of<AbstractDevice, device>::value, "Device supports cpu and gpu only!");
	static_assert(is_integral<DType>::value || is_floating_point<DType>::value, "DType supports integral and float point only!");
	static const bool _isCPU = device::_isCPU;
	static const bool _isGPU = device::_isGPU;

	static const size_t _kDim = 0;
	const bool _isLeaf;
	
	Shape<0> _shape;
	size_t _stride;
	DType *_ptr;

	bool _isUpdated;
	
	XMATRIX_INLINE Tensor(bool isLeaf = true) : _isLeaf(isLeaf), _ptr(NULL), _isUpdated(false) {}

	XMATRIX_INLINE virtual ~Tensor() { FreeMem(); }

	XMATRIX_INLINE void Input(DType * pData, Shape<0> shape = Shape0()) {
		Invalid();
		AllocMem(shape);
		if (_isCPU) {
			memcpy(_ptr, pData, _shape.getSize() * sizeof(DType));
		}
	}

	XMATRIX_INLINE void AllocMem(Shape<0> shape = Shape0()) {
		Invalid();
		FreeMem();
		_shape = shape;
		_stride = shape.SubShape().getSize();
		if (_isCPU)
			_ptr = (DType*)calloc(_shape.getSize(), sizeof(DType));
	}

	XMATRIX_INLINE void FreeMem() {
		if (_ptr != NULL) {
			if (_isCPU)
				free(_ptr);
		}
		_ptr = NULL;
	}

	XMATRIX_INLINE virtual void Update() {
		_isUpdated = true;
	}

	XMATRIX_INLINE virtual void Invalid() {
		_isUpdated = false;
	}
}; 

template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE ostream &operator<<(ostream &os, Tensor<device, dimension, DType> &t) {
	if (t._ptr != NULL) {
		Tensor<device, dimension - 1, DType> &t0 = t[0];
		t0.Update();
		os << "[" << t0;
		for (size_t i = 1; i < t._shape[0]; i++) {
			Tensor<device, dimension - 1, DType> &ti = t[i];
			ti.Update();
			os << ", ";
			if (t._kDim > 1) {
				os << endl << " ";
			}
			os << ti;
		}
		os << "]";
	}
	return os;
}

template<typename device, typename DType>
XMATRIX_INLINE ostream &operator<<(ostream &os, Tensor<device, 0, DType> &t) {
	if (t._ptr != NULL) {
		os << t._ptr[0];
	}
	return os;
}

template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct UnaryDeducedTensor : public Tensor<device_dest, dimension_dest, DType_dest> {
	Tensor<device_src, dimension_src, DType_src> &_src;

	XMATRIX_INLINE UnaryDeducedTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: Tensor<device_dest, dimension_dest, DType_dest>(false), _src(src) {}

	XMATRIX_INLINE virtual void Update() {
		Tensor<device_dest, dimension_dest, DType_dest>::Update();
		_src.Update();
	}

	XMATRIX_INLINE virtual void Invalid() {
		Tensor<device_dest, dimension_dest, DType_dest>::Invalid();
		_src.Invalid();
	}
};

template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct BinaryDeducedTensor : public Tensor<device_dest, dimension_dest, DType_dest> {
	Tensor<device_lhs, dimension_lhs, DType_lhs> &_lhs;
	Tensor<device_rhs, dimension_rhs, DType_rhs> &_rhs;

	XMATRIX_INLINE BinaryDeducedTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs) 
		: Tensor<device_dest, dimension_dest, DType_dest>(false), _lhs(lhs), _rhs(rhs) {}

	XMATRIX_INLINE virtual void Update() {
		Tensor<device_dest, dimension_dest, DType_dest>::Update();
		_lhs.Update();
		_rhs.Update();
	}

	XMATRIX_INLINE virtual void Invalid() {
		Tensor<device_dest, dimension_dest, DType_dest>::Invalid();
		_lhs.Invalid();
		_rhs.Invalid();
	}
};

/**
* Add Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct AddTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE AddTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Minus Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct MinusTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE MinusTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Multiple Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct MultipleTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE MultipleTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Dot Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct DotTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE DotTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Divide Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct DivideTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE DivideTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Transpose Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct TransposeTensor 
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE TransposeTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Subscript Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct SubscriptTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {

	const double _index;
	
	XMATRIX_INLINE SubscriptTensor(Tensor<device_src, dimension_src, DType_src> &src, size_t index) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src), _index(index) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Exponential Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct ExponentialTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE ExponentialTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Log Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct LogTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE LogTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Log10 Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct Log10Tensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE Log10Tensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Sqrt Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct SqrtTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE SqrtTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Power Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct PowerTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {

	const double _exp;
	
	XMATRIX_INLINE PowerTensor(Tensor<device_src, dimension_src, DType_src> &src, double exp) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src), _exp(exp) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Abs Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct AbsTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE AbsTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Floor Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct FloorTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE FloorTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Ceil Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct CeilTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE CeilTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Round Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct RoundTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE RoundTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* GreaterThanTensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct GreaterThanTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE GreaterThanTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* EqualTensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct EqualTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE EqualTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* AndTensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct AndTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE AndTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* OrTensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_lhs, size_t dimension_lhs, typename DType_lhs,
	typename device_rhs, size_t dimension_rhs, typename DType_rhs>
struct OrTensor 
	: public BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs> {

	XMATRIX_INLINE OrTensor(
		Tensor<device_lhs, dimension_lhs, DType_lhs> &lhs, 
		Tensor<device_rhs, dimension_rhs, DType_rhs> &rhs)
	: BinaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_lhs, dimension_lhs, DType_lhs, device_rhs, dimension_rhs, DType_rhs>
		(lhs, rhs) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* NotTensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct NotTensor 
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {

	XMATRIX_INLINE NotTensor(Tensor<device_src, dimension_src, DType_src> &src)
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Sign Tensor: return 1, -1 or 0
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct SignTensor 
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {

	XMATRIX_INLINE SignTensor(Tensor<device_src, dimension_src, DType_src> &src)
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Sum Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct SumTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE SumTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
* Mean Tensor
*/
template<typename device_dest, size_t dimension_dest, typename DType_dest,
	typename device_src, size_t dimension_src, typename DType_src>
struct MeanTensor
	: public UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src> {
	
	XMATRIX_INLINE MeanTensor(Tensor<device_src, dimension_src, DType_src> &src) 
		: UnaryDeducedTensor<device_dest, dimension_dest, DType_dest, device_src, dimension_src, DType_src>(src) {
			cerr << "Not supported yet!" << endl;
			assert(false);
	}
};

/**
*
*/

} // namespace xmatrix
#endif // MATRIX_TENSOR_H_