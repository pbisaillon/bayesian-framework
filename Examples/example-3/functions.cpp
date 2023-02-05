#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

const double AMPLITUDE 	= 400.0;
const double FREQ 		=		0.5;
const double MASS			= 	1.0;
const double DAMPING	=		3.0;

/** Generating model for example 1 **/
extern "C" colvec massSpringDamper(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	double kc = parameters[1];
	double sigmag = parameters[2];
	double amplitude = AMPLITUDE;
	double forceFreq = FREQ;

	double x1 = state[0];
	double x2 = state[1];

	/* Record the white-noise */
	double force = sigmag*std::sqrt(dt)*randn();

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*(-k*x1/m - kc*x1*x1*x1/m - c*x2/m + amplitude/m*cos(2.0 * datum::pi * forceFreq * time)) + force/m; //add the force
	temp[2] = force;                                                                                                                         	//
	return temp;
}

/*************************************************************
************************** Example1 Model ********************
*************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	double kc = parameters[1];
	double amplitude = AMPLITUDE;
	double forceFreq = FREQ;

	colvec temp = state;
	double x1 = state[0];
	double x2 = state[1];

	/* Explicit */
	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt * (-k/m*x1 - kc*x1*x1*x1/m - c/m * x2 + amplitude/m * cos(2.0 * datum::pi * forceFreq * time));

	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	double kc = parameters[1];
	double x = state[0];
	return {{1.0,dt},{-dt*(k/m + 3.0*kc/m*x*x), 1.0-dt*c/m}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double m = MASS;
	double sigma = parameters[2];
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma/m}};
}

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}


colvec model2( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	double amplitude = AMPLITUDE;
	double forceFreq = FREQ;

	colvec temp = state;
	double x1 = state[0];
	double x2 = state[1];

	/* Explicit */
	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt * (-k/m*x1 - c/m * x2 + amplitude/m * cos(2.0 * datum::pi * forceFreq * time));
	return temp;
}

// Declare all the jacobians for EKF
mat dfdx2 (const colvec & state, const double dt, const colvec & parameters ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	return {{1.0,dt},{-dt*(k/m), 1.0-dt*c/m}};
}

mat dfde2 (const colvec&, const double dt, const colvec & parameters ) {
	double m = MASS;
	double sigma = parameters[1];
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma/m}};
}

/*************************************************************
************************************ Stochastic models
*************************************************************/

colvec model1s( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double m = MASS;
	double c = DAMPING;
	double k = parameters[0];
	double kc = parameters[1];
	double sigma = parameters[2];
	double amplitude = AMPLITUDE;
	double forceFreq = FREQ;

	double x1 = state[0];
	double x2 = state[1];

	/* Record the white-noise */
	double force = sigma*std::sqrt(dt)*randn();

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*(-k*x1/m - kc*x1*x1*x1/m - c*x2/m + amplitude/m*cos(2.0 * datum::pi * forceFreq * time)) + force/m;                                                                                                                      	//
	return temp;
}
//Stochastic (adds the noise, for EnKF)
colvec _hs( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0] + std::sqrt(covariance.at(0,0))*randn();
	return h;
}

double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = d[0]-state[0];
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0);
}

/*************************************************************
************************************ Build all the statespaces
*************************************************************/
extern "C" statespace m1 	= statespace(model1, dfdx, 	dfde, _h, dhdx, dhde, 2, 1);
extern "C" statespace m2 	= statespace(model2, dfdx2, dfde2, _h, dhdx, dhde, 2, 1);
extern "C" statespace m1s = statespace(model1s, _hs, logfunc, 2, 1);
