#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

/* Example of a mass-spring-damper system with parallel springs k1 and k2
* where k2 spring snaps at time t
*
*/

/** Generating model 1st order ODE */
const double mass = 1.0;
const double amplitude = 0.0;
const double freq = 2.0 * 3.1416 * 1.5;
const double trueC = 0.8;
extern "C" colvec msd(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	double k1 = parameters[0];
	double k2 = parameters[1];
	double c = parameters[2];
	double sigma = parameters[3];
	double timeSnaps = parameters[4];
	double K;

	if (time < timeSnaps) {
		K = k1 + k2;
	} else {
		K = k1;
	}
	//std::cout << "Stiffness: " << K << std::endl;
	double uk = state[0];
	double vk = state[1];

	/* Record the white-noise */
	double force = sigma/mass*std::sqrt(dt)*randn() + dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + K*uk) + force;	                                                                                                                  	//
	return temp;
}


/****************************************************************************************
************************* Estimate K , c and sigma **************************************
*****************************************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double K = parameters[0];
	double c = parameters[1];
	double sigma = parameters[2];

	double uk = state[0];
	double vk = state[1];

	/* Record the white-noise */
	double force = dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + K*uk) + force;	                                                                                                                  	//
	return temp;
}


colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	double K = parameters[0];
	double c = parameters[1];

	return {{1.0,dt},{-dt*K/mass, 1.0-dt/mass*c}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[2];
	//std::cout << "Sigma = " << sigma << std::endl;
	return {{0.0,0.0},{0.0,sigma*std::sqrt(dt)/mass}};
}

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

/****************************************************************************************
************************* Estimate k1,k2, c, sigma and t snaps **************************************
*****************************************************************************************/

static double staticTime;

colvec model2( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double k1 = parameters[0];
	double k2 = parameters[1];
	double c = parameters[2];
	double tsnaps = parameters[3];
	double sigma = parameters[4];


	double K;
	staticTime = time;
	if (time < tsnaps) {
		K = k1 + k2;
	} else {
		K = k1;
	}

	double uk = state[0];
	double vk = state[1];

	/* Record the white-noise */
	double force = dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + K*uk) + force;	                                                                                                                  	//
	return temp;
}

// Declare all the jacobians for EKF
mat dfdx2 (const colvec & state, const double dt, const colvec & parameters ) {
	double k1 = parameters[0];
	double k2 = parameters[1];
	double c = parameters[2];
	double tsnaps = parameters[3];
	double K;
	//not yet implemented. Using a workaround
	if (staticTime < tsnaps) {
		K = k1 + k2;
	} else {
		K = k1;
	}

	return {{1.0,dt},{-dt*K/mass, 1.0-dt/mass*c}};
}

mat dfde2 (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[4];
	return {{0.0,0.0},{0.0,sigma*std::sqrt(dt)/mass}};
}

/****************************************************************************************
************************* Estimate c, sigma as static and k as time varying parameter ***
*****************************************************************************************/

colvec model3( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double c = parameters[0];

	double uk = state[0];
	double vk = state[1];
	double Kk = state[2];

	/* Record the white-noise */
	double force = dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + Kk*uk) + force;
	temp[2] = Kk; //No update here	                                                                                                                  	//
	return temp;
}

// Declare all the jacobians for EKF
mat dfdx3 (const colvec & state, const double dt, const colvec & parameters ) {
	double c = parameters[0];
	double uk = state[0];
	double vk = state[1];
	double Kk = state[2];

	return {{1.0,dt,0.0},
					{-dt*Kk/mass, 1.0-dt/mass*c, -dt*uk/mass},
					{0.0, 0.0, 1.0}};
}

mat dfde3 (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[1];
	double gamma = 1.0;
	//std::cout << "Sigma = " << sigma << std::endl;
	return {{0.0,0.0,0.0},
					{0.0,sigma*std::sqrt(dt)/mass, 0.0},
					{0.0, 0.0, gamma*std::sqrt(dt)} };
}

mat dhdx3 (const colvec& , const colvec & parameters ) {
	return {1.0,0.0, 0.0};
}

mat dfde4 (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[1];
	double gamma = parameters[2];
	//std::cout << "Sigma = " << sigma << std::endl;
	return {{0.0,0.0,0.0},
					{0.0,sigma*std::sqrt(dt)/mass, 0.0},
					{0.0, 0.0, gamma*std::sqrt(dt)} };
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
/////////////// Model 1 + colored noise

colvec model5( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double K = parameters[0];
	double c = parameters[1];
	double tau		= parameters[2];


	double uk = state[0];
	double vk = state[1];
	double qk = state[2];

	/* Record the white-noise */
	double force = dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + K*uk) + force + dt*qk/mass;
	temp[2] = std::exp(-dt/tau) * qk;	                                                                                                                  	//
	return temp;
}

// Declare all the jacobians for EKF
mat dfdx5 (const colvec & state, const double dt, const colvec & parameters ) {
	double K = parameters[0];
	double c = parameters[1];
	double tau		= parameters[2];

	return {{1.0,dt,0.0},{-dt*K/mass, 1.0-dt/mass*c , dt/mass},{0.0, 0.0, std::exp(-dt/tau)}};
}

mat dfde5 (const colvec&, const double dt, const colvec & parameters ) {
	double tau = parameters[2];
	double sigma = parameters[3];

	//not scaled: var sigmaSqr
	//Scaled 2sigmaSqr/tau
	//std::cout << "Sigma = " << sigma << std::endl;
	//return {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,std::sqrt(sigma*sigma*tau*0.5*(1.0-std::exp(-2.0*dt/tau)))}};
	return {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,std::sqrt(2.0*dt/tau)*sigma}};
}


/****************************************************************************************
************************* Model 6 Estimate c, sigma as static and k as time varying parameter
************************* also estimates K(0) and gamma
*****************************************************************************************/

colvec model6( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double c = parameters[0];

	double uk = state[0];
	double vk = state[1];
	double Kk = state[2];

	/* Record the white-noise */
	double force = dt*amplitude/mass*cos(freq * time);
	colvec temp = state;

	if (time == 0 ) {
		Kk = parameters[3];
	}

	temp[0] = uk + vk * dt;
	temp[1] = vk - dt/mass*(c*vk + Kk*uk) + force;
	temp[2] = Kk; //No update here	                                                                                                                  	//
	return temp;
}


/*************************************************************
************************************ Build all the statespaces
*************************************************************/
extern "C" statespace m2 = statespace(model1, dfdx, 	dfde, _h, dhdx, dhde, 2, 1);
extern "C" statespace m1 = statespace(model2, dfdx2, 	dfde2, _h, dhdx, dhde, 2, 1);
extern "C" statespace m3 = statespace(model3, dfdx3, 	dfde3, _h, dhdx3, dhde, 3, 1);
extern "C" statespace m4 = statespace(model3, dfdx3, 	dfde4, _h, dhdx3, dhde, 3, 1);
extern "C" statespace m6 = statespace(model6, dfdx3, 	dfde4, _h, dhdx3, dhde, 3, 1);

extern "C" statespace m5 = statespace(model5, dfdx5, 	dfde5, _h, dhdx3, dhde, 3, 1);
