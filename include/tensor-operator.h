#ifndef XMATRIX_TENSOR_OPERATOR_H_
#define XMATRIX_TENSOR_OPERATOR_H_

#include "common.h"
#include "tensor.h"

namespace xmatrix {

template<typename device, size_t dimension, typename DType>
struct Tensor_Wrapper {
	Tensor<device, dimension, DType> * _tensor;

	XMATRIX_INLINE Tensor_Wrapper() {
		_tensor = new Tensor<device, dimension, DType>();
	}

	XMATRIX_INLINE Tensor_Wrapper(const Tensor_Wrapper<device, dimension, DType> &tensor) {
		_tensor = tensor._tensor;
	}

	XMATRIX_INLINE Tensor_Wrapper(Tensor<device, dimension, DType> tensor) {
		_tensor = &tensor;
	}

	XMATRIX_INLINE Tensor_Wrapper(Tensor<device, dimension, DType>* tensor) {
		_tensor = tensor;
	}

	XMATRIX_INLINE Tensor_Wrapper<device, dimension, DType> &operator=(const Tensor_Wrapper<device, dimension, DType> &tensor) {
		_tensor->FreeMem();
		delete _tensor;
		_tensor = tensor._tensor;

		return *this;
	}

	XMATRIX_INLINE Tensor_Wrapper<device, dimension, DType> &operator=(const Tensor<device, dimension, DType> &tensor) {
		_tensor->FreeMem();
		delete _tensor;
		_tensor = &tensor;

		return *this;
	}

	XMATRIX_INLINE Tensor_Wrapper<device, dimension, DType> &operator=(const Tensor<device, dimension, DType> *tensor) {
		_tensor->FreeMem();
		delete _tensor;
		_tensor = tensor;

		return *this;
	}

	XMATRIX_INLINE virtual ~Tensor_Wrapper() {
		_tensor->FreeMem();
		delete _tensor;
	}

	XMATRIX_INLINE void Load(DType * pData, const Shape<dimension> &shape) {
		_tensor->FreeMem();
		_tensor->Input(pData, shape);
	}

	XMATRIX_INLINE void Free() {
		_tensor->FreeMem();
	}

	XMATRIX_INLINE void Update() {
		if (_tensor != NULL)
			_tensor->Update();
	}

}; // tensor_wrapper

template<typename device, typename DType>
struct Scalar {
	typedef Tensor_Wrapper<device, 0, DType> type;
};

template<typename device, typename DType>
struct Vector {
	typedef Tensor_Wrapper<device, 1, DType> type;

	XMATRIX_INLINE static Shape<1> LoadCSV(Tensor_Wrapper<device, 1, DType> &dest, std::string filename, char delimiter = ',') {
		std::ifstream fin(filename.c_str());
		std::string line;
		std::vector<DType> buffer;

		size_t length = 0;

		while(std::getline(fin, line)) {
			std::stringstream lin(line);
			std::string cell;
			while(std::getline(lin, cell, delimiter)) {
				if (cell.length() > 0) {
					buffer.push_back((DType)atof(cell.c_str()));
					length += 1;
				}
			}
		}

		Shape<1> s = Shape1(length);
		dest._tensor->AllocMem(s);
		if (dest._tensor->_isCPU)
			std::copy(buffer.begin(), buffer.end(), dest._tensor->_ptr);
		return s;
	}
};

template<typename device, typename DType>
struct Matrix {
	typedef Tensor_Wrapper<device, 2, DType> type;
	
	XMATRIX_INLINE static Shape<2> LoadCSV(Tensor_Wrapper<device, 2, DType> &dest, std::string filename, char delimiter = ',') {
		std::ifstream fin(filename.c_str());
		std::string line;
		std::vector<std::vector<DType>> buffer;

		size_t col = 0, row = 0;

		while(std::getline(fin, line)) {
			std::stringstream lin(line);
			std::string cell;
			std::vector<DType> array;
			while(std::getline(lin, cell, delimiter)) {
				if (cell.length() > 0)
					array.push_back((DType)atof(cell.c_str()));
			}
			col = (array.size() > col)? array.size() : col;
			row += 1;
			buffer.push_back(array);
		}

		Shape<2> s = Shape2(row, col);
		dest._tensor->AllocMem(s);

		for(size_t i=0; i<row; i++) {
			if (dest._tensor->_isCPU) {
				std::copy(buffer[i].begin(), buffer[i].end(), dest._tensor->_ptr + i * col);
			}
		}

		return s;
	}
};

template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE ostream &operator<<(ostream &os, const Tensor_Wrapper<device, dimension, DType> &tensor) {
	os << *tensor._tensor;
	return os;
}
/**
* Add Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {

	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())>(
			new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())>(
			new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>())>(
			new AddTensor<device, dimension, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
				device, dimension, DType_rhs, device, 0, DType_lhs>(*(rhs._tensor), *(lhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>())> &operator+(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>())>(
			new AddTensor<device, 0, decltype(declval<DType_lhs>() + declval<DType_rhs>()), 
				device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() + declval<DType_Param>())> &operator+(
	Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src + (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() + declval<DType_Param>())> &operator+(
	DType param, Tensor_Wrapper<device, dimension, DType_Param> &src) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return *t + src;
}

/**
* Minus Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {

	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())>(
			new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())>(
			new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>())>(
			new MinusTensor<device, dimension, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
				device, dimension, DType_rhs, device, 0, DType_lhs>(*(rhs._tensor), *(lhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>())> &operator-(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>())>(
			new MinusTensor<device, 0, decltype(declval<DType_lhs>() - declval<DType_rhs>()), 
				device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() - declval<DType_Param>())> &operator-(
	Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src - (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() - declval<DType_Param>())> &operator-(
	DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) - src;
}

template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, DType> &operator-(Tensor_Wrapper<device, dimension, DType> &src) {
	return src * (-1);
}

/**
* Multiple Operator
*/
template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, 1, DType_lhs> &lhs, Tensor_Wrapper<device, 2, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, 1, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, 1, DType_lhs, device, 2, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, 1, DType_lhs> &lhs, Tensor_Wrapper<device, 1, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, 1, DType_lhs, device, 1, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, 2, DType_lhs> &lhs, Tensor_Wrapper<device, 2, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, 2, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, 2, DType_lhs, device, 2, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, dimension, DType_rhs, device, 0, DType_lhs>(*(rhs._tensor), *(lhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &operator*(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new MultipleTensor<device, 0, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() * declval<DType_Param>())> &operator*(
	Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src * (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() * declval<DType_Param>())> &operator*(
	DType param, Tensor_Wrapper<device, dimension, DType_Param> &src) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) * src;
}

/**
* Divide Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())>(
			new DivideTensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>())>(
			new DivideTensor<device, dimension, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
				device, 0, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>())> &operator/(
	Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	
	Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>())> *t 
		= new Tensor_Wrapper<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>())>(
			new DivideTensor<device, 0, decltype(declval<DType_lhs>() / declval<DType_rhs>()), 
				device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() / declval<DType_Param>())> &operator/(
	Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src / (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType>() / declval<DType_Param>())> &operator/(
	DType param, Tensor_Wrapper<device, dimension, DType_Param> &src) {

	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) / src;
}

/**
* Not Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new NotTensor<device, dimension, int, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Sign Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new SignTensor<device, dimension, int, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* And Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator&&(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new AndTensor<device, dimension, int, device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator&&(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new AndTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator&&(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new AndTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator&&(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	Tensor_Wrapper<device, 0, int> *t  
		= new Tensor_Wrapper<device, 0, int>(
			new AndTensor<device, 0, int, device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator&&(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src && (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator&&(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) && src;
}

/**
* Or Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator||(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new OrTensor<device, dimension, int, device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator||(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new OrTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator||(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t  
		= new Tensor_Wrapper<device, dimension, int>(
			new OrTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator||(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	Tensor_Wrapper<device, 0, int> *t  
		= new Tensor_Wrapper<device, 0, int>(
			new OrTensor<device, 0, int, device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator||(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src || (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator||(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) || src;
}

/**
* XOR Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator^(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return (!lhs && rhs) || (lhs && !rhs);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator^(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	return (!src && param) || (src && !param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator^(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	return (!src && param) || (src && !param);
}

template<typename device, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator^(Tensor_Wrapper<device, 0, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	return (!src && param) || (src && !param);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator^(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);

	return (!src && (*t)) || (src && !(*t));
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator^(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (!src && (*t)) || (src && !(*t));
}

/**
* Equal Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator==(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator==(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator==(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator==(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	Tensor_Wrapper<device, 0, int> *t 
		= new Tensor_Wrapper<device, 0, int>(
			new EqualTensor<device, 0, int, device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator==(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);

	return src == (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator==(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return (*t) == src;
}

/**
* Not Equal Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!=(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return !(*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!=(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return !(*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!=(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new EqualTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return !(*t);
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator!=(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	Tensor_Wrapper<device, 0, int> *t 
		= new Tensor_Wrapper<device, 0, int>(
			new EqualTensor<device, 0, int, device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return !(*t);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!=(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);
	return src != (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator!=(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);

	return (*t) != src;
}

/**
* Greater than operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new GreaterThanTensor<device, dimension, int, device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>(Tensor_Wrapper<device, dimension, DType> &src, Tensor_Wrapper<device, 0, DType_Param> &param) {
	Tensor_Wrapper<device, dimension, int> *t 
		= new Tensor_Wrapper<device, dimension, int>(
			new GreaterThanTensor<device, dimension, int, device, dimension, DType, device, 0, DType_Param>(*(src._tensor), *(param._tensor)));
	return *t;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>(Tensor_Wrapper<device, 0, DType_Param> &param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, int> *t
		= new Tensor_Wrapper<device, dimension, int>(
			new GreaterThanTensor<device, dimension, int, device, 0, DType, device, dimension, DType_Param>(*(param._tensor), *(src._tensor)));
	return *t;
}

template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator>(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	Tensor_Wrapper<device, 0, int> *t 
		= new Tensor_Wrapper<device, 0, int>(
			new GreaterThanTensor<device, 0, int, device, 0, DType_lhs, device, 0, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));
	return *t;
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);

	return src > (*t);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, DType_Param> *t 
		= new Tensor_Wrapper<device, 0, DType_Param>(new Tensor<device, 0, DType_Param>());
	t->_tensor->Input(&param);

	return (*t) > src;
}

/**
* Less Than Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return rhs > lhs;
}
	
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return rhs > lhs;
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return rhs > lhs;
}
	
template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator<(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return rhs > lhs;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	return param > src;
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	return src > param;
}

/**
* Not Less Than operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>=(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return !(lhs < rhs);
}
	
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>=(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return !(lhs < rhs);
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>=(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return !(lhs < rhs);
}
	
template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator>=(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return !(lhs < rhs);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>=(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	return !(src < param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator>=(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	return !(param < src);
}

/**
* Not Greater Than operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<=(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return !(lhs > rhs);
}
	
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<=(Tensor_Wrapper<device, dimension, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return !(lhs > rhs);
}

template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<=(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
	return !(lhs > rhs);
}
	
template<typename device, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, 0, int> &operator<=(Tensor_Wrapper<device, 0, DType_lhs> &lhs, Tensor_Wrapper<device, 0, DType_rhs> &rhs) {
	return !(lhs > rhs);
}
	
template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<=(Tensor_Wrapper<device, dimension, DType> &src, DType_Param param) {
	return !(src > param);
}

template<typename device, size_t dimension, typename DType, typename DType_Param>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &operator<=(DType_Param param, Tensor_Wrapper<device, dimension, DType> &src) {
	return !(param > src);
}

namespace op {
/**
* Dot Operator
*/
template<typename device, size_t dimension, typename DType_lhs, typename DType_rhs>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> &Dot(
	Tensor_Wrapper<device, dimension, DType_lhs> &lhs, 
	Tensor_Wrapper<device, dimension, DType_rhs> &rhs) {
		
	Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())> *t = 
		new Tensor_Wrapper<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>())>(
			new DotTensor<device, dimension, decltype(declval<DType_lhs>() * declval<DType_rhs>()), 
				device, dimension, DType_lhs, device, dimension, DType_rhs>(*(lhs._tensor), *(rhs._tensor)));

	return *t;
}


/**
* Transpose Operator
*/
template<typename device, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 2, DType> &Transpose(Tensor_Wrapper<device, 2, DType> &src) {
	Tensor_Wrapper<device, 2, DType> *t 
		= new Tensor_Wrapper<device, 2, DType>(
			new TransposeTensor<device, 2, DType, device, 2, DType>(*(src._tensor)));
	return *t;
}

template<typename device, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 2, DType> &Transpose(Tensor_Wrapper<device, 1, DType> &src) {
	Tensor_Wrapper<device, 2, DType> *t 
		= new Tensor_Wrapper<device, 2, DType>(
			new TransposeTensor<device, 2, DType, device, 1, DType>(*(src._tensor)));
	return *t;
}

/**
* Exponential Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, double> &Exp(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new ExponentialTensor<device, dimension, double, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Log Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, double> &Log(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new LogTensor<device, dimension, double, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Log10 Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, double> &Log10(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new Log10Tensor<device, dimension, double, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Sqrt Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, double> &Sqrt(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new SqrtTensor<device, dimension, double, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Power Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, double> &Pow(Tensor_Wrapper<device, dimension, DType> &src, double exp) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new PowerTensor<device, dimension, double, device, dimension, DType>(*(src._tensor), exp));
	return *t;
}

/**
* Abs Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, DType> &Abs(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new AbsTensor<device, dimension, DType, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Floor Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &Floor(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new FloorTensor<device, dimension, int, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Ceil Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &Ceil(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new CeilTensor<device, dimension, int, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Round Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, dimension, int> &Round(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, dimension, double> *t 
		= new Tensor_Wrapper<device, dimension, double>(
			new RoundTensor<device, dimension, int, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Sum Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 0, DType> &Sum(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, double> *t 
		= new Tensor_Wrapper<device, 0, double>(
			new SumTensor<device, 0, DType, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* Mean Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 0, DType> &Mean(Tensor_Wrapper<device, dimension, DType> &src) {
	Tensor_Wrapper<device, 0, double> *t 
		= new Tensor_Wrapper<device, 0, double>(
			new MeanTensor<device, 0, DType, device, dimension, DType>(*(src._tensor)));
	return *t;
}

/**
* All Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 0, DType> &All(Tensor_Wrapper<device, dimension, DType> &src) {
	return op::Mean(src > 0) == 1;
}

/**
* Any Operator
*/
template<typename device, size_t dimension, typename DType>
XMATRIX_INLINE Tensor_Wrapper<device, 0, DType> &Any(Tensor_Wrapper<device, dimension, DType> &src) {
	return op::Sum(src > 0) >= 1;
}

} // namespace op

} // namespace xmatrix

#endif // XMATRIX_TENSOR_OPERATOR_H_