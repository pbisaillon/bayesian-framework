#include <mcmc.hpp>
#include <pdf.hpp>
#include <samples.hpp>
#include <filters.hpp>
#include <statespace.hpp>
#include <wrapper.hpp>
#include <armadillo>
#include <cmath>
#include <iomanip>      // std::setprecision
#include <functional>
#include <config.hpp>
#include <bayesianPosterior.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h> //to load dynamic library

using namespace arma;
colvec mu = {-1.0 , 10.0};
mat sigma = {{2.0,1.0},{1.0,4.0}};
mat sigmaInv = inv(sigma);
double logGaussian( const colvec& x) {
	return - 0.5 * as_scalar( trans(x - mu)*sigmaInv*(x-mu) );
}

int main(int argc, char* argv[]) {

	/* MPI */
	MPI::Init ( argc, argv );
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ();

	/* Set the seed to a random value */
	arma_rng::set_seed_random();


	/* Creating the bayesian posterior object */
	bayesianPosterior bp = bayesianPosterior( logGaussian );

	/* Configuration of MCMC chain */
	mat identity = eye<mat>(2,2);
	colvec zeroVec =  zeros<colvec>(2);
	int N = 50000;

	mcmc mychainA = mcmc(N, identity  , bp  );
	mcmc mychainB = mcmc(N, identity  , bp  );
	mcmc mychainC = mcmc(N, identity  , bp  );
	mcmc mychainD = mcmc(N, identity  , bp  );

	mychainA.setSartingPoint( zeroVec );
	mychainB.setSartingPoint( zeroVec );
	mychainC.setSartingPoint( zeroVec );
	mychainD.setSartingPoint( zeroVec );

	/*  Chain A - Metropolis Hastings. No burn-in */
	mychainA.metropolisHastings();

	/*  Chain B - 1000 AP - Metropolis Hastings. */
	mychainB.adaptiveProposal( 200, 5 );
	mychainB.metropolisHastings();


	/*  Chain C - 1000 AP - AM. */
	mychainC.adaptiveProposal( 200, 5 );
	mychainC.adaptiveMetropolis(500);

	/*  Chain D - 1000 AP - DRAM. */
	mychainD.adaptiveProposal( 200, 5 );
	mychainD.DRAM(200, 0.5, 0.1);

	mychainA.print("." , "chainA" );
	mychainB.print("." , "chainB" );
	mychainC.print("." , "chainC" );
	mychainD.print("." , "chainD" );

	//Estimating from the chain
	double estimateA = exp(mychainA.logEvidence( mychainA.getMAP() , N , 5 ));
	double estimateB = exp(mychainB.logEvidence( mychainB.getMAP() , N , 5 ));
	double estimateC = exp(mychainC.logEvidence( mychainC.getMAP() , N , 5 ));
	double estimateD = exp(mychainD.logEvidence( mychainD.getMAP() , N , 5 ));

	system("clear");


	//Quadrature
	//Any lower order the covariance can not be approximated correctly
	GaussHermite GH = GaussHermite(bp , 4, 2);
	double estimateGH = exp( GH.quadrature(1.0e-10, 15) );

	double actual = 2.0*datum::pi * std::sqrt(det(sigma));

	std::cout << "Evidence computation from chain using Chib-Jeliazkov." << std::endl;
	std::cout << "Analytical = " << actual << std::endl;

	std::cout << "Estimate (Metropolis-Hastings, no burn-in) = " << estimateA << std::endl;
	std::cout << "Error = " << std::abs(actual - estimateA )/actual * 100.0 << "% (with " << N << " samples)" << std::endl << std::endl;

	std::cout << "Estimate (Metropolis-Hastings, AP burn-in) = " << estimateB << std::endl;
	std::cout << "Error = " << std::abs( actual - estimateB )/actual * 100.0 << "% (with " << N << " samples)" << std::endl << std::endl;

	std::cout << "Estimate (AP then AM) = " << estimateC << std::endl;
	std::cout << "Error = " << std::abs( actual - estimateC )/actual * 100.0 << "% (with " << N << " samples)" << std::endl << std::endl;

	std::cout << "Estimate (AP then DRAM) = " << estimateD << std::endl;
	std::cout << "Error = " << std::abs(actual - estimateD )/actual * 100.0 << "% (with " << N << " samples)" << std::endl << std::endl;

	std::cout << "Estimate (Quad) = " << estimateGH << std::endl;
	std::cout << "Error = " << std::abs(actual - estimateGH )/actual * 100.0 << "%" << std::endl << std::endl;


	MPI::Finalize();
	return 0;
}
