#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

/** Generating model for example 1 **/
extern "C" colvec vdposcillator(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	double mu = parameters[0];
	double amp = parameters[1];
	double omega = parameters[2];
	double sigma = parameters[3];

	double x1 = state[0];
	double x2 = state[1];

	/* Record the white-noise */
	double force = sigma*std::sqrt(dt)*randn();

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*mu*(1.0-x1*x1)*x2 - dt*x1 + dt*amp*sin(omega * time) + force;
	temp[2] = force;	                                                                                                                  	//
	return temp;
}

/*************************************************************
************************* Van Der Pol Oscillator Model*******
*************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double mu = parameters[0];
	double amp = parameters[1];
	double omega = parameters[2];

	double x1 = state[0];
	double x2 = state[1];

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*mu*(1.0-x1*x1)*x2 - dt*x1 + dt*amp*sin(omega * time);
	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	double mu = parameters[0];

	double x1 = state[0];
	double x2 = state[1];
	return {{1.0,dt},{dt*(mu*(-2.0*x1)*x2-1.0), 1.0+dt*mu*(1.0-x1*x1)}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[3];
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma}};
}

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

/*************************************************************
****** Van Der Pol Oscillator Model (Stochastic) *************
*************************************************************/
colvec model1s( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double mu = parameters[0];
	double amp = parameters[1];
	double omega = parameters[2];
	double sigma = parameters[3];

	double x1 = state[0];
	double x2 = state[1];

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*mu*(1.0-x1*x1)*x2 - dt*x1 + dt*amp*sin(omega * time) + sigma*std::sqrt(dt)*randn();
	return temp;
}

colvec hs = zeros<colvec>(1);
colvec _hs( const colvec & state, const mat & covariance, const colvec & parameters ) {
	hs[0] = state[0] + std::sqrt(covariance.at(0,0))*randn();
	return hs;
}

//Log function
double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = d[0]-state[0];
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0);
}

/*************************************************************
************************************ Build all the statespaces
*************************************************************/
extern "C" statespace mekf = statespace(model1, dfdx, 	dfde, _h, dhdx, dhde, 2, 1);
extern "C" statespace menkf = statespace(model1s, _hs, logfunc, 2, 1);
extern "C" statespace mpf = statespace(model1s, _h, logfunc, 2, 1);
