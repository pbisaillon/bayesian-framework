#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

/** Generating model for example 1 **/
extern "C" colvec sineSignal(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	//Equation 3.32 ekfukf toolbox documentation
	double q1 = parameters[0];
	double q2 = parameters[1];

	double theta = state[0];
	double omega = state[1];
	double a 		 = state[2];

	/* Record the white-noise */
	//Neglect higher order terms (dt^2 and dt^3)
	double force1 = q1*std::sqrt(dt)*randn();
	double force2 = q2*std::sqrt(dt)*randn();

	colvec temp = state;

	temp[0] = theta + dt * omega;
	temp[1] = omega + force1;
	temp[2] = a + force2;
	temp[3] = 0.0;	                                                                                                                  	//
	return temp;
}

/*************************************************************
************************* Sine Signal Oscillator Model*******
*************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {

	double theta = state[0];
	double omega = state[1];
	double a 		 = state[2];

	colvec temp = state;

	temp[0] = theta + dt * omega;
	temp[1] = omega;
	temp[2] = a;
	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[2]*std::sin(state[0]);
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	return {{1.0,dt,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	//Q matrix will be {{0,0,0},{0,1,0},{0,0,1}}
	double q1 = parameters[0];
	double q2 = parameters[1];
	return {{0.0,0.0,0.0},{0.0,std::sqrt(dt*q1),0.0},{0.0,0.0,std::sqrt(dt*q2)}};
}

mat dhdx (const colvec& state, const colvec & parameters ) {
	double theta = state[0];
	double omega = state[1];
	double a 		 = state[2];
	return {a*std::cos(theta),0.0, std::sin(theta)};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

/*************************************************************
****** Van Der Pol Oscillator Model (Stochastic) *************
*************************************************************/
colvec model1s( const colvec & state, const double dt, double time, const colvec & parameters ) {
	//Equation 3.32 ekfukf toolbox documentation
	double q1 = parameters[0];
	double q2 = parameters[1];

	double theta = state[0];
	double omega = state[1];
	double a 		 = state[2];

	/* Record the white-noise */
	//Neglect higher order terms (dt^2 and dt^3)
	double force1 = q1*std::sqrt(dt)*randn();
	double force2 = q2*std::sqrt(dt)*randn();

	colvec temp = state;

	temp[0] = theta + dt * omega;
	temp[1] = omega + force1;
	temp[2] = a + force2;                                                                                                                  	//
	return temp;
}

colvec hs = zeros<colvec>(1);
colvec _hs( const colvec & state, const mat & covariance, const colvec & parameters ) {
	hs[0] = state[2]*std::sin(state[0]) + std::sqrt(covariance.at(0,0))*randn();
	return hs;
}

//Log function
double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = d[0]-state[2]*std::sin(state[0]);
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0);
}

/*************************************************************
************************************ Build all the statespaces
*************************************************************/
extern "C" statespace mekf = statespace(model1, dfdx, 	dfde, _h, dhdx, dhde, 3, 1);
//extern "C" statespace menkf = statespace(model1s, _hs, logfunc, 4, 1); Will it work? need to extend the sate ...
extern "C" statespace mpf = statespace(model1s, _h, logfunc, 3, 1);
