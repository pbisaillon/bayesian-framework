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

###Plotting state estimation results
ekf = np.loadtxt('./Case00/EKF-state-estimation.dat')
enkf = np.loadtxt('./Case00/ENKF-state-estimation.dat')
pf = np.loadtxt('./Case00/PF-state-estimation.dat')
#pfenkf = np.loadtxt('./Case00/PFENKF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(ekf[:,0], ekf[:,1], 'r')
plt.plot(enkf[:,0], enkf[:,1], 'g')
plt.plot(pf[:,0], pf[:,1], 'b')
plt.legend(['True','EKF','ENKF','PF'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,1], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_x_estimate.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(true[-1,:], true[1,:], 'k')
plt.plot(ekf[:,0], ekf[:,3], 'r')
plt.plot(enkf[:,0], enkf[:,3], 'g')
plt.plot(pf[:,0], pf[:,3], 'b')
plt.legend(['True','EKF','ENKF','PF'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case00_v_estimate.eps", format='eps', dpi=1000)
plt.clf()



init_plotting(12,18,800)
true = np.loadtxt('./Case01/true.dat')
data = np.loadtxt('./Case01/data.dat')
datat = np.loadtxt('./Case01/timedata.dat')
plt.xlabel('Time')
plt.ylabel('x')

plt.plot(datat, data, 'ro')
plt.ylabel('x')
plt.plot(true[-1,:], true[0,:], 'k')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case01_data.eps", format='eps', dpi=1000)
plt.clf()

###Plotting state estimation results
ekf = np.loadtxt('./Case01/EKF-state-estimation.dat')
enkf = np.loadtxt('./Case01/ENKF-state-estimation.dat')
pf = np.loadtxt('./Case01/PF-state-estimation.dat')
#pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

plt.plot(true[-1,:], true[0,:], 'k')
plt.plot(ekf[:,0], ekf[:,1], 'r')
plt.plot(enkf[:,0], enkf[:,1], 'g')
plt.plot(pf[:,0], pf[:,1], 'b')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.legend(['True','EKF','ENKF','PF'], loc='best')
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case01_x_estimate.eps", format='eps', dpi=1000)
plt.clf()

plt.plot(true[-1,:], true[1,:], 'k')
plt.plot(ekf[:,0], ekf[:,3], 'r')
plt.plot(enkf[:,0], enkf[:,3], 'g')
plt.plot(pf[:,0], pf[:,3], 'b')
plt.legend(['True','EKF','ENKF','PF'], loc='best')
#plt.plot(pfenkf[:,0], pfenkf[:,3], 'b--')
#plt.legend(['True','EKF','ENKF','PF', 'PFENKF'])
plt.grid(False)
plt.tight_layout()
plt.savefig("./figs/case01_v_estimate.eps", format='eps', dpi=1000)
plt.clf()
