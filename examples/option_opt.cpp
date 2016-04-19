#define XMATRIX_USE_CUDA 0
#define XMATRIX_USE_MKL 0

#include "xmatrix.h"
#include <cmath>

using namespace xmatrix;

int main(int argc, char* argv[]) {
	// Model Parameters
	Matrix<cpu, double>::type delta, gamma, rho, theta, vega;
	Vector<cpu, double>::type iv, price_prob, vol_iv, vol_RfIn;
	Vector<cpu, int>::type sp;

	// Model Variables
	Vector<cpu, double>::type qty;

	Vector<cpu, double>::type net_delta, net_gamma, net_rho, net_theta, net_vega;
	net_delta = qty * delta;
	net_gamma = qty * gamma;
	net_rho = qty * rho;
	net_theta = qty * theta;
	net_vega = qty * vega;

	Vector<cpu, double>::type HPPriceVol, HPPriceChange, HPPriceChangeSQ;
	HPPriceVol = op::Exp(iv * sqrt(3.0 / 365)) - 1;
	HPPriceChange = op::Dot(sp, HPPriceVol);
	HPPriceChangeSQ = op::Pow(HPPriceChange, 2);	
	
	Vector<cpu, double>::type Delta_Effect, Gamma_Effect, Rho_Effect, Theta_Effect, Vega_Effect;
	Delta_Effect = op::Dot(-op::Abs(net_delta), HPPriceChange);
	Gamma_Effect = op::Dot(net_gamma, HPPriceChangeSQ) / 2;
	Theta_Effect = net_theta * 3.0;
	Vega_Effect  = op::Dot(-op::Abs(net_vega), vol_iv) * sqrt(3.0 * 5 / 7);
	Rho_Effect   = op::Dot(-op::Abs(net_rho), vol_RfIn) * sqrt(3.0 * 5 / 7);
	
	Vector<cpu, double>::type DTRRR, VTRRR, RTRRR;
	DTRRR = op::Dot(Delta_Effect + Gamma_Effect, 1.0 / Theta_Effect);
	VTRRR = op::Dot(Vega_Effect, 1.0 / Theta_Effect);
	RTRRR = op::Dot(Rho_Effect, 1.0 / Theta_Effect);
	
	double a = 0.5, b = 0.5, c = 0;
	Scalar<cpu, double>::type opt;
	opt = (a * DTRRR + b * VTRRR + c * RTRRR) * price_prob;

	Matrix<cpu, double>::LoadCSV(delta, "data/delta.csv");
	Matrix<cpu, double>::LoadCSV(gamma, "data/gamma.csv");
	Matrix<cpu, double>::LoadCSV(rho, "data/rho.csv");
	Matrix<cpu, double>::LoadCSV(theta, "data/theta.csv");
	Matrix<cpu, double>::LoadCSV(vega, "data/vega.csv");
	
	Vector<cpu, double>::LoadCSV(iv, "data/iv.csv");
	Vector<cpu, int>::LoadCSV(sp, "data/price.csv");
	Vector<cpu, double>::LoadCSV(vol_iv, "data/vol_iv.csv");
	Vector<cpu, double>::LoadCSV(vol_RfIn, "data/vol_RfIn.csv");
	Vector<cpu, double>::LoadCSV(price_prob, "data/price_prob.csv");

	double quality[] = {
		1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1};
	qty.Load(quality, Shape1(35));
	opt.Update();
	std::cout << opt << std::endl << std::endl;

	return 0;
}