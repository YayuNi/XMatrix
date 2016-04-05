#ifndef XMATRIX_TENSOR_OPERATOR_H_
#define XMATRIX_TENSOR_OPERATOR_H_

#include "common.h"
#include "tensor.h"

namespace xmatrix {

/**
* Add Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {

	Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t
		= new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, dimension, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
			device, dimension, DType_rhs, device, 0, DType_lhs>(rhs, lhs);
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new AddTensor<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
			device, 0, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor<device, dimension, DType_lhs> &src, DType_rhs param) {

	Scalar<device, DType_rhs>::type *p = new Scalar<device, DType_rhs>::type();
	p->Input(&param);
	return src + *p;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	DType_lhs param, Tensor<device, dimension, DType_rhs> &src) {

	Scalar<device, DType_lhs>::type *p = new Scalar<device, DType_lhs>::type();
	p->Input(&param);
	return src + *p;
}

/**
* Minus Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {

	Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t
		= new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, dimension, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
			device, dimension, DType_rhs, device, 0, DType_lhs>(rhs, lhs);
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new MinusTensor<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
			device, 0, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor<device, dimension, DType_lhs> &src, DType_rhs param) {

	Scalar<device, DType_rhs>::type *p = new Scalar<device, DType_rhs>::type();
	p->Input(&param);
	return src - *p;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	DType_lhs param, Tensor<device, dimension, DType_rhs> &src) {

	Scalar<device, DType_lhs>::type *p = new Scalar<device, DType_lhs>::type();
	p->Input(&param);
	return src - *p;
}

/**
* Multiple Operator
*/
template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, 1, DType_lhs> &lhs, Tensor<device, 2, DType_rhs> &rhs) {
	
	Tensor<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new MultipleTensor<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
			device, 1, DType_lhs, device, 2, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, 2, DType_lhs> &lhs, Tensor<device, 2, DType_rhs> &rhs) {
	
	Tensor<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new MultipleTensor<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
			device, 2, DType_lhs, device, 2, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new MultipleTensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new MultipleTensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
			device, dimension, DType_rhs, device, 0, DType_lhs>(rhs, lhs);
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new MultipleTensor<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
			device, 0, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor<device, dimension, DType_lhs> &src, DType_rhs param) {

	Scalar<device, DType_rhs>::type *p = new Scalar<device, DType_rhs>::type();
	p->Input(&param);
	return src * (*p);
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	DType_lhs param, Tensor<device, dimension, DType_rhs> &src) {

	Scalar<device, DType_lhs>::type *p = new Scalar<device, DType_lhs>::type();
	p->Input(&param);
	return src * (*p);
}

/**
* Divide Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor<device, dimension, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new DivideTensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
			device, dimension, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, dimension, DType_rhs> &rhs) {
	
	Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new DivideTensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
			device, 0, DType_lhs, device, dimension, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor<device, 0, DType_lhs> &lhs, Tensor<device, 0, DType_rhs> &rhs) {
	
	Tensor<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new DivideTensor<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
			device, 0, DType_lhs, device, 0, DType_rhs>(lhs, rhs);
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor<device, dimension, DType_lhs> &src, DType_rhs param) {

	Scalar<device, DType_rhs>::type *p = new Scalar<device, DType_rhs>::type();
	p->Input(&param);
	return src / (*p);
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	DType_lhs param, Tensor<device, dimension, DType_rhs> &src) {

	Scalar<device, DType_lhs>::type *p = new Scalar<device, DType_lhs>::type();
	p->Input(&param);
	return (*p) / src;
}

/**
* Not Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new NotTensor<device, dimension, int, device, dimension, DType>(src);
	return *t;
}

/**
* And Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator&&(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new AndTensor<device, dimension, int, device, dimension, DType, device, dimension, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator&&(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new AndTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator&&(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new AndTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator&&(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, 0, int> *t = new AndTensor<device, 0, int, device, 0, DType, device, 0, DType_Param>(src, param);
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator&&(Tensor<device, dimension, DType> &src, DType_Param param) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src && (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator&&(DType_Param param, Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src && (*t);
}

/**
* Or Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator||(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new OrTensor<device, dimension, int, device, dimension, DType, device, dimension, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator||(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new OrTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator||(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new OrTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator||(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, 0, int> *t = new OrTensor<device, 0, int, device, 0, DType, device, 0, DType_Param>(src, param);
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator||(Tensor<device, dimension, DType> &src, DType_Param param) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src || (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator||(DType_Param param, Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src || (*t);
}

/**
* XOR Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator^(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	return (!src && param) || (src && !param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator^(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return (!src && param) ^ (src && !param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator^(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	return (!src && param) ^ (src && !param);
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator^(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return (!src && param) ^ (src && !param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator^(Tensor<device, dimension, DType> &src, DType_Param param) {
	return (!src && param) ^ (src && !param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator^(DType_Param param, Tensor<device, dimension, DType> &src) {
	return (!src && param) ^ (src && !param);
}

/**
* Equal Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator==(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, dimension, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator==(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator==(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator==(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, 0, int> *t = new EqualTensor<device, 0, int, device, 0, DType, device, 0, DType_Param>(src, param);
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator==(Tensor<device, dimension, DType> &src, DType_Param param) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src == (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator==(DType_Param param, Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src == (*t);
}

/**
* Not Equal Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!=(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, dimension, DType_Param>(src, param);
	return !(*t);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!=(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return !(*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!=(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return !(*t);
}
	
template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator!=(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, 0, int> *t = new EqualTensor<device, 0, int, device, 0, DType, device, 0, DType_Param>(src, param);
	return !(*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!=(Tensor<device, dimension, DType> &src, DType_Param param) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src != (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator!=(DType_Param param, Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src != (*t);
}

/**
* Greater than operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new GreaterThanTensor<device, dimension, int, device, dimension, DType, device, dimension, DType_Param>(src, param);
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, dimension, int> *t = new GreaterThanTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new GreaterThanTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(src, param);
	return *t;
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator>(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	Tensor<device, 0, int> *t = new GreaterThanTensor<device, 0, int, device, 0, DType, device, 0, DType_Param>(src, param);
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>(Tensor<device, dimension, DType> &src, DType_Param param) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src > (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>(DType_Param param, Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType_Param> *t = new Tensor<device, 0, DType_Param>();
	t->Input(&param);
	return src > (*t);
}

/**
* Less Than Operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	return param > src;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return param > src;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	return src > param;
}
	
template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator<(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return param > src;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<(Tensor<device, dimension, DType> &src, DType_Param param) {
	return param > src;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<(DType_Param param, Tensor<device, dimension, DType> &src) {
	return src > param;
}

/**
* Not Less Than operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>=(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	return !(src < param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>=(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return !(src < param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>=(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	return !(param < src);
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator>=(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return !(src < param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>=(Tensor<device, dimension, DType> &src, DType_Param param) {
	return !(src < param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator>=(DType_Param param, Tensor<device, dimension, DType> &src) {
	return !(param < src);
}

/**
* Not Greater Than operator
*/
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<=(Tensor<device, dimension, DType> &src, Tensor<device, dimension, DType_Param> &param) {
	return !(src > param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<=(Tensor<device, dimension, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return !(src > param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<=(Tensor<device, 0, DType_Param> &param, Tensor<device, dimension, DType> &src) {
	return !(param > src);
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, 0, int> &operator<=(Tensor<device, 0, DType> &src, Tensor<device, 0, DType_Param> &param) {
	return !(src > param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<=(Tensor<device, dimension, DType> &src, DType_Param param) {
	return !(src > param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor<device, dimension, int> &operator<=(DType_Param param, Tensor<device, dimension, DType> &src) {
	return !(param > src);
}

namespace op {
/**
* Transpose Operator
*/
template<typename device, typename DType>
XMATRIX_INLINE Tensor<device, 2, DType> &Transpose(Tensor<device, 2, DType> &src) {
	Tensor<device, 2, DType> *t = new TransposeTensor<device, 2, DType, device, 2, DType>(src);
	return *t;
}

template<typename device, typename DType>
XMATRIX_INLINE Tensor<device, 2, DType> &Transpose(Tensor<device, 1, DType> &src) {
	Tensor<device, 2, DType> *t = new TransposeTensor<device, 2, DType, device, 1, DType>(src);
	return *t;
}

/**
* Exponential Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, double> &Exp(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, double> *t = new ExponentialTensor<device, dimension, double, device, dimension, DType>(src);
	return *t;
}

/**
* Log Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, double> &Log(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, double> *t = new LogTensor<device, dimension, double, device, dimension, DType>(src);
	return *t;
}

/**
* Log10 Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, double> &Log10(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, double> *t = new Log10Tensor<device, dimension, double, device, dimension, DType>(src);
	return *t;
}

/**
* Sqrt Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, double> &Sqrt(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, double> *t = new SqrtTensor<device, dimension, double, device, dimension, DType>(src);
	return *t;
}

/**
* Power Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, double> &Pow(Tensor<device, dimension, DType> &src, double exp) {
	Tensor<device, dimension, double> *t = new PowerTensor<device, dimension, double, device, dimension, DType>(src, exp);
	return *t;
}

/**
* Abs Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, DType> &Abs(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, DType> *t = new AbsTensor<device, dimension, DType, device, dimension, DType>(src);
	return *t;
}

/**
* Floor Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, int> &Floor(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new FloorTensor<device, dimension, int, device, dimension, DType>(src);
	return *t;
}

/**
* Ceil Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, int> &Ceil(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new CeilTensor<device, dimension, int, device, dimension, DType>(src);
	return *t;
}

/**
* Round Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, dimension, int> &Round(Tensor<device, dimension, DType> &src) {
	Tensor<device, dimension, int> *t = new RoundTensor<device, dimension, int, device, dimension, DType>(src);
	return *t;
}

/**
* Sum Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, 0, DType> &Sum(Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType> *t = new SumTensor<device, 0, DType, device, dimension, DType>(src);
	return *t;
}

/**
* Mean Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, 0, DType> &Mean(Tensor<device, dimension, DType> &src) {
	Tensor<device, 0, DType> *t = new MeanTensor<device, 0, DType, device, dimension, DType>(src);
	return *t;
}

/**
* All Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, 0, DType> &All(Tensor<device, dimension, DType> &src) {
	return Mean(src > 0) == 1;
}

/**
* Any Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor<device, 0, DType> &Any(Tensor<device, dimension, DType> &src) {
	return Sum(src > 0) >= 1;
}

} // namespace op

} // namespace xmatrix

#endif // XMATRIX_TENSOR_OPERATOR_H_