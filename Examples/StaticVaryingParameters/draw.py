import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import sys
import getopt
sys.path.append('../../visualization')
from visualisation import *
import re

##Creating figs folder if it doesn't already exists
if not os.path.isdir("./figs"):
    os.makedirs("figs")

init_plotting(12,18,800)
true = np.loadtxt('./Case00/true.dat')
data = np.loadtxt('./Case00/data.dat')
datat = np.loadtxt('./Case00/timedata.dat')
plt.xlabel('Time')
plt.ylabel('x')

plt.plot(datat, data, 'ro')
plt.ylabel('x')
plt.plot(true[-1,:], true[0,:], 'k')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_data.eps", format='eps', dpi=1000)
plt.clf()

###Plotting samples of a and sigma
samples = np.loadtxt('./Case00/chains/StaticParameters-posterior.dat')
plt.plot(samples[0,::5], samples[1,::5], 'k.')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_samples.eps", format='eps', dpi=1000)
plt.clf()

samplesAonly = np.loadtxt('./Case00/chains/StaticParametersAonly-posterior.dat')
plt.plot(samples[0,::5], samples[1,::5], 'k.')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_samples_aonly.eps", format='eps', dpi=1000)
plt.clf()

###Plotting marginal pdfs of a and sigma

x,y = get_marginal_pdf(samples[0,:], 500, bw = 0.4)
plt.plot(x, y, 'k')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a.eps", format='eps', dpi=1000)
plt.clf()

x,y = get_marginal_pdf(samples[1,:], 500, bw = 0.4)
plt.plot(x, y, 'k')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_sigma.eps", format='eps', dpi=1000)
plt.clf()

x,y = get_marginal_pdf(samplesAonly[:], 500, bw = 0.4)
plt.plot(x, y, 'k')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_aOnly.eps", format='eps', dpi=1000)
plt.clf()

#def get_1D_kde(samples, numpoints, minx=None , maxx=None, bandwidth = 0.2, linear_approx = 0):
#def get_marginal_pdf(samples, numpoints, bw = 0.2, xl=None, xr=None):



###Plotting state estimation results
m1 = np.loadtxt('./Case00/TimeVaryingParametersB-state-estimation.dat')
m1WithSigma  = np.loadtxt('./Case00/TimeVaryingParametersWithSigmaB-state-estimation.dat')
m1enkf = np.loadtxt('./Case00/TimeVaryingParametersEnkfB-state-estimation.dat')
m1pf = np.loadtxt('./Case00/TimeVaryingParametersPfB-state-estimation.dat')
#pf = np.loadtxt('./Case00/PF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(m1[:,0], m1[:,1], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,1], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,1], 'b')
plt.plot(m1pf[:,0], m1pf[:,1], 'g')
plt.legend(['True','EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(m1[:,0], m1[:,3], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,3], 'b')
plt.plot(m1pf[:,0], m1pf[:,3], 'g')
plt.plot(m1[:,0], m1[:,3]+3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1[:,0], m1[:,3]-3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]+3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]-3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.legend(['EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a_estimate.eps", format='eps', dpi=1000)
plt.clf()



###Plotting state estimation results
m1 = np.loadtxt('./Case00/TimeVaryingParametersC-state-estimation.dat')
m1WithSigma  = np.loadtxt('./Case00/TimeVaryingParametersWithSigmaC-state-estimation.dat')
m1enkf = np.loadtxt('./Case00/TimeVaryingParametersEnkfC-state-estimation.dat')
m1pf = np.loadtxt('./Case00/TimeVaryingParametersPfC-state-estimation.dat')
#pf = np.loadtxt('./Case00/PF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(m1[:,0], m1[:,1], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,1], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,1], 'b')
plt.plot(m1pf[:,0], m1pf[:,1], 'g')
plt.legend(['True','EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate2.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(m1[:,0], m1[:,3], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,3], 'b')
plt.plot(m1pf[:,0], m1pf[:,3], 'g')
plt.plot(m1[:,0], m1[:,3]+3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1[:,0], m1[:,3]-3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]+3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]-3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.legend(['EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a_estimate2.eps", format='eps', dpi=1000)
plt.clf()

#
#
#
#
m1 = np.loadtxt('./Case00/TimeVaryingParametersD-state-estimation.dat')
m1WithSigma  = np.loadtxt('./Case00/TimeVaryingParametersWithSigmaD-state-estimation.dat')
m1enkf = np.loadtxt('./Case00/TimeVaryingParametersEnkfD-state-estimation.dat')
m1pf = np.loadtxt('./Case00/TimeVaryingParametersPfD-state-estimation.dat')
#pf = np.loadtxt('./Case00/PF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(m1[:,0], m1[:,1], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,1], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,1], 'b')
plt.plot(m1pf[:,0], m1pf[:,1], 'g')
plt.legend(['True','EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate3.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(m1[:,0], m1[:,3], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,3], 'b')
plt.plot(m1pf[:,0], m1pf[:,3], 'g')
plt.plot(m1[:,0], m1[:,3]+3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1[:,0], m1[:,3]-3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]+3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]-3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.legend(['EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a_estimate3.eps", format='eps', dpi=1000)
plt.clf()

#
#
#
#
m1 = np.loadtxt('./Case00/TimeVaryingParametersE-state-estimation.dat')
m1WithSigma  = np.loadtxt('./Case00/TimeVaryingParametersWithSigmaE-state-estimation.dat')
m1enkf = np.loadtxt('./Case00/TimeVaryingParametersEnkfE-state-estimation.dat')
m1pf = np.loadtxt('./Case00/TimeVaryingParametersPfE-state-estimation.dat')
#pf = np.loadtxt('./Case00/PF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(m1[:,0], m1[:,1], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,1], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,1], 'b')
plt.plot(m1pf[:,0], m1pf[:,1], 'g')
plt.legend(['True','EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate4.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(m1[:,0], m1[:,3], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,3], 'b')
plt.plot(m1pf[:,0], m1pf[:,3], 'g')
plt.plot(m1[:,0], m1[:,3]+3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1[:,0], m1[:,3]-3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]+3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]-3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.legend(['EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a_estimate4.eps", format='eps', dpi=1000)
plt.clf()


#
#
#
#
m1 = np.loadtxt('./Case00/TimeVaryingParametersF-state-estimation.dat')
m1WithSigma  = np.loadtxt('./Case00/TimeVaryingParametersWithSigmaF-state-estimation.dat')
m1enkf = np.loadtxt('./Case00/TimeVaryingParametersEnkfF-state-estimation.dat')
m1pf = np.loadtxt('./Case00/TimeVaryingParametersPfF-state-estimation.dat')
#pf = np.loadtxt('./Case00/PF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(m1[:,0], m1[:,1], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,1], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,1], 'b')
plt.plot(m1pf[:,0], m1pf[:,1], 'g')
plt.legend(['True','EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate5.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(m1[:,0], m1[:,3], 'r')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3], 'y')
plt.plot(m1enkf[:,0], m1enkf[:,3], 'b')
plt.plot(m1pf[:,0], m1pf[:,3], 'g')
plt.plot(m1[:,0], m1[:,3]+3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1[:,0], m1[:,3]-3.0*np.sqrt(m1[:,3]), 'r--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]+3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.plot(m1WithSigma[:,0], m1WithSigma[:,3]-3.0*np.sqrt(m1WithSigma[:,3]), 'y--')
plt.legend(['EKF (sigma = 0)', 'EKF (sigma = 0.01)', 'ENKF (sigma = 0.01)', 'PF (sigma = 0.01)'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_a_estimate5.eps", format='eps', dpi=1000)
plt.clf()
