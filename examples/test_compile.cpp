#define XMATRIX_USE_CUDA 0
#define XMATRIX_USE_MKL 0

#include "xmatrix.h"
#include "model.h"
#include "random-cpu.h"

using namespace xmatrix;

void testShape() {
	cout << "****************** TEST CASE: Shape *****************" << endl;

	Shape<0> s0 = Shape0();
	cout << s0 << endl;

	Shape<1> s1 = Shape1(10);
	cout << s1 << endl;
	cout << s1.SubShape() << endl;

	Shape<2> s2 = Shape2(10,5);
	cout << s2 << endl;
	cout << s2.SubShape() << endl;
	cout << s2.SubShape().SubShape() << endl;

	cout << endl;
}

void testTensorAdd() {
	cout << "***** TEST CASE: Tensor Addition between DTypes *****" << endl;

	Vector<cpu, int>::type &a = Vector<cpu, int>::type(),  &b = Vector<cpu, int>::type();
	Vector<cpu, float>::type &c = Vector<cpu, float>::type();
	Vector<cpu, int>::type &d = a - b;
	Vector<cpu, float>::type &e = a + b - c;
	
	size_t length = 3;
	int data_a[] = {1,2,3}, data_b[] = {4,5,6};
	float data_c[] = {4.5, 5.5, 6.5};
	a.Input(data_a, Shape1(length));
	b.Input(data_b, Shape1(length));
	c.Input(data_c, Shape1(length));
	d.Update();
	e.Update();
	cout << d << endl << e << endl;

	cout << endl;
}

void testTensorMultiple() {
	cout << "******** TEST CASE: Vector = Vector x Matrix ********" << endl;

	Tensor<cpu, 2, int> &a = Matrix<cpu, int>::type();
	Tensor<cpu, 1, float> &b = Vector<cpu, float>::type();
	Tensor<cpu, 1, float> &c = b * a;
	
	size_t nrow = 2, ncol = 3;
	int data_a[] = {1,2,3,4,5,6};
	float data_b[] = {7.5,8.5};
	a.Input(data_a, Shape2(nrow, ncol));
	b.Input(data_b, Shape1(nrow));
	c.Update();

	cout << a << endl;
	cout << b << endl;
	cout << c << endl << endl;
}

void testTensorMultiple2() {
	cout << "******** TEST CASE: Matrix = Matrix x Matrix ********" << endl;
	size_t i = 3, j = 2;
	Tensor<cpu, 2, float> &a = Matrix<cpu, float>::type();
	Tensor<cpu, 2, double> &b = Matrix<cpu, double>::type();
	Tensor<cpu, 2, double> &c = a * b, &d = b * a;
	float data_a[] = {1,2,3,4,5,6};
	double data_b[] = {6,5,4,3,2,1};
	a.Input(data_a, Shape2(i,j));
	b.Input(data_b, Shape2(j,i));
	c.Update();
	d.Update();
	
	cout << c << endl << d << endl << endl;
}

void testTensorMultiple3() {
	cout << "******** TEST CASE: Scalar = Scalar x Scalar ********" << endl;
	Tensor<cpu, 0, int> &a = Scalar<cpu, int>::type();
	Tensor<cpu, 0, double> &b = Scalar<cpu, double>::type();
	Tensor<cpu, 0, double> &c = a*b;
	int data_a = 10;
	double data_b = 5.5;
	a.Input(&data_a);
	b.Input(&data_b);
	c.Update();
	cout << c << endl << endl;
	
	cout << "******** TEST CASE: Scalar = Scalar x Matrix ********" << endl;
	Tensor<cpu, 0, float> &d = Scalar<cpu, float>::type();
	Tensor<cpu, 2, int> &e = Matrix<cpu, int>::type();
	Tensor<cpu, 2, float> &f = d*e, &g = e*d;

	float data_d = 5;
	int data_e[] = {1,2,3,4};
	d.Input(&data_d);
	e.Input(data_e, Shape2(2,2));
	f.Update();
	g.Update();

	cout << f << endl << g << endl;
	cout << endl;
}

void testTensorArithmetic() {
	cout << "************ TEST CASE: Tensor Airthmetic ***********" << endl;

	Tensor<cpu, 0, float> &a = Scalar<cpu, float>::type(), &h = Scalar<cpu, float>::type(), &w = Scalar<cpu, float>::type();
	Tensor<cpu, 1, double> &b = Vector<cpu, double>::type(), &c = Vector<cpu, double>::type();
	Tensor<cpu, 2, int> &e = Matrix<cpu, int>::type();
	Tensor<cpu, 1, double> &r = h + a * b / h + c * e - w;
	Matrix<cpu, double>::type &f = 1.0 / e;

	float data_a = 10, data_h = 5, data_w = 20;
	double data_b[] = {1,2}, data_c[] = {3,4};
	int data_e[] = {5,6,7,8};
	a.Input(&data_a);
	h.Input(&data_h);
	w.Input(&data_w);
	b.Input(data_b, Shape1(2));
	c.Input(data_c, Shape1(2));
	e.Input(data_e, Shape2(2,2));
	r.Update();
	f.Update();

	cout << r << endl;
	cout << f << endl;

	Tensor<cpu, 0, float> &x = a + h;
	x.Update();
	cout << x << endl << endl;
}

void testTensorOutput() {
	
	cout << "********** TEST CASE: Tensor Output Format **********" << endl;
	Tensor<cpu, 0, int> &a = Scalar<cpu, int>::type();
	int data_a = 10;
	a.Input(&data_a);
	cout << a << endl;

	Tensor<cpu, 1, int> &b = Vector<cpu, int>::type();
	int data_b[] = {1,2,3,4,5};
	b.Input(data_b, Shape1(5));
	cout << b << endl;

	Tensor<cpu, 2, int> &c = Matrix<cpu, int>::type(), &e = op::Transpose(c);
	int data_c[] = {1,2,3,4};
	c.Input(data_c, Shape2(2,2));
	e.Update();
	cout << c << endl << e << endl;

	Shape<3> s;
	s[0] = 2; s[1] = 2; s[2] = 2;
	Tensor<cpu, 3, int> &d = Tensor<cpu, 3, int>();
	int data_d[] = {1,2,3,4,5,6,7,8};
	d.Input(data_d, s);
	cout << d << endl << endl;
}

void testOps() {
	cout << "************ TEST CASE: Tensor Operations ***********" << endl;
	Tensor<cpu, 2, float> &c = Matrix<cpu, float>::type(), &e = op::Abs(c);
	Tensor<cpu, 2, double> &d = op::Sqrt(op::Abs(op::Round(c)));
	float data_c[] = {1.5f,-2.2f,3.7f,-4.5f};
	c.Input(data_c, Shape2(2,2));
	d.Update();
	cout << d << endl << endl;	
}

void testLogic() {
	cout << "************** TEST CASE: Tensor Logics *************" << endl;

	Matrix<cpu, float>::type &a = Matrix<cpu, float>::type();
	Matrix<cpu, int>::type &b = (a == 0.6f);
	Scalar<cpu, float>::type &s = op::Sum(a);
	float data_a[] = {0.2f, 0.4f, 0.6f, 0.8f};
	a.Input(data_a, Shape2(2,2));
	b.Update();
	s.Update();
	cout << b << endl << s << endl << endl;
}

void testRandom() {
	cout << "************ TEST CASE: Random Generator ************" << endl;

	Matrix<cpu, double>::type &x = Matrix<cpu, double>::type();
	Shape<2> s = Shape2(5, 10);

	Random<cpu> r;
	r.GaussianInit(x, s);
	cout << x << endl << endl;
}

int main() {
	testShape();
	testTensorAdd();
	testTensorMultiple();
	testTensorMultiple2();
	testTensorMultiple3();
	testTensorArithmetic();
	testTensorOutput();
	testOps();
	testLogic();
	testRandom();

	int a = 0, b = 1, &c = a;
	c = b;
	cout << a << endl;

	return 0;
}