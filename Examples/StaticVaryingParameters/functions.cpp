#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

/* See 8.4 An Example of Constant Parameter Indentification p. 115
*	 in Kalman Filtering with Real-Time Applications by Chui and Chen
*
*/

/** Generating model 1st order ODE */
extern "C" colvec FirstOrder(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	double alpha = parameters[0];
	double sigma = parameters[1];

	double xk = state[0];

	/* Record the white-noise */
	double force = sigma*std::sqrt(dt)*randn();
	colvec temp = state;

	temp[0] = (1.0 - alpha * dt)*xk + force;
	temp[1] = force;	                                                                                                                  	//
	return temp;
}

/****************************************************************************************
*************************Treat alpha as a time varying parameter (no model noise) ********
*****************************************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double xk = state[0];
	double ak = state[1];

	colvec temp = state;

	temp[0] = (1.0 - ak * dt)*xk; // + force;
	temp[1] = ak;

	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

colvec model1s( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double xk = state[0];
	double ak = state[1];

	colvec temp = state;
	double sigma = parameters[0];
	double gamma = parameters[1];

	temp[0] = (1.0 - ak * dt)*xk + sigma*std::sqrt(dt)*randn(); // + force;
	temp[1] = ak + gamma*std::sqrt(dt)*randn();

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

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {

	double xk = state[0];
	double ak = state[1];
	return {{1.0 - ak * dt,  -xk*dt},{0.0, 1.0}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[0];
	double gamma = parameters[1];
	return {{sigma*std::sqrt(dt),0.0},{0.0,gamma*std::sqrt(dt)}};
}


mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

/*************************************************************
****** Treat alpha and sigma as constant parameter *******************
*************************************************************/
colvec model2( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double a = parameters[0];
	double xk = state[0];

	colvec temp = state;

	temp[0] = (1.0 - a * dt)*xk; // + force;

	return temp;
}

mat dfdx2 (const colvec & state, const double dt, const colvec & parameters ) {
	double a = parameters[0];
	return {1.0 - a * dt};
}

mat dfde2 (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[1];
	return {sigma*std::sqrt(dt)};
}

mat dfde2a (const colvec&, const double dt, const colvec & parameters ) {
	//double sigma = parameters[1];
	double sigma = 0.01;
	return {sigma*std::sqrt(dt)};
}

mat dhdx2 (const colvec& , const colvec & parameters ) {
	return {1.0};
}

/*************************************************************
************************************ Build all the statespaces
*************************************************************/
extern "C" statespace m1 = statespace(model1, dfdx, 	dfde, _h, dhdx, dhde, 2, 1);
extern "C" statespace m1enkf = statespace(model1s, _hs, logfunc, 2, 1);
extern "C" statespace m2 = statespace(model2, dfdx2, 	dfde2, _h, dhdx2, dhde, 1, 1);
extern "C" statespace m2a = statespace(model2, dfdx2, 	dfde2a, _h, dhdx2, dhde, 1, 1);
