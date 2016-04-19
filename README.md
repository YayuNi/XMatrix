# XMatrix
A C++ symbolic math library for optimizing and machine learning.

Example: Matrix manipulate
------
```C++

#include "xmatrix.h"

using namespace xmatrix;

int main(int argc, char* argv[]) {

	// Row vector
	Vector<cpu, double>::type x, y, b;

	// Matrix
	Matrix<cpu, double>::type w;

	// Deduce formula
	y = x * w + b;

	// Load data for x, w and b
	double data_x[] = {1.0, 2.0, 3.0};
	double data_b[] = {-1.0, -2.0};

	double data_w[] = {2.0, 1.0,
	                   3.0, 2.0,
			   1.0, 3.0};
	x.Load(data_x, Shape1(3));
	b.Load(data_b, Shape1(2));
	w.Load(data_w, Shape2(3,2));

	// Deduce value of y
	y.Update();
	std::cout << y << std::endl;

	return 0;
}

```
