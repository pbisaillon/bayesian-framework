import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import getopt
sys.path.append('../../visualization')
from visualisation import *
import re

init_plotting()
true = np.loadtxt('./Results/true.dat')
data = np.loadtxt('./Results/data.dat')
datat = np.loadtxt('./Results/timedata.dat')
plt.xlabel('Time [s]')
plt.ylabel('x [m]')

plt.plot(datat, data, 'ro')
plt.ylabel('x [m]')
plt.plot(true[-1,:], true[0,:], 'k')
plt.grid(False)
plt.tight_layout()
#plt.savefig("linear-case-a-data.pdf", format='pdf', dpi=1000)
plt.savefig("./figs/Results_data.eps", format='eps', dpi=1000)
plt.clf()

a = np.loadtxt('./Results/Model1-000.dat')
b = np.loadtxt('./Results/Model1-001.dat')
c = np.loadtxt('./Results/Model1-002.dat')
d = np.loadtxt('./Results/Model1-003.dat')
chainm1 = np.concatenate((a,b,c,d), axis=0)
#chainm1 = np.loadtxt('./Results/Model1.dat')
Xm1,Ym1 = get_marginal_pdf(chainm1[:,1], 500, 0.2)

init_plotting()
plt.xlabel('$k$')
plt.ylabel('pdf')
plt.plot(Xm1, Ym1)

plt.vlines(2000.0,0,max(Ym1), linestyles=[(0,(9,3,4,4))], colors='k')
plt.legend(['$\\mathcal{M}_1$', 'Truth'], loc = "upper right")
plt.tight_layout()
#plt.savefig("linear-case-a-k.pdf", format='pdf', dpi=1000)
plt.savefig("./figs/k.eps", format='eps', dpi=1000)
plt.clf()

Xm1,Ym1 = get_marginal_pdf(chainm1[:,2], 500, 0.2)
init_plotting()
plt.xlabel('$k_c$')
plt.ylabel('pdf')
plt.plot(Xm1, Ym1)
plt.vlines(-500.0,0,max(Ym1), linestyles=[(0,(9,3,4,4))], colors='k')
plt.legend(['$\\mathcal{M}_1$', 'Truth'], loc = "upper right")
plt.tight_layout()
#plt.savefig("linear-case-a-k.pdf", format='pdf', dpi=1000)
plt.savefig("./figs/kc.eps", format='eps', dpi=1000)
plt.clf()

Xm1,Ym1 = get_marginal_pdf(chainm1[:,3], 500, 0.2)

init_plotting()
plt.xlabel('$\\sigma$')
plt.ylabel('pdf')
plt.plot(Xm1, Ym1)
plt.vlines(1.0,0,max(Ym1), linestyles=[(0,(9,3,4,4))], colors='k')
plt.legend(['$\\mathcal{M}_1$', 'Truth'], loc = "upper right")
plt.tight_layout()
#plt.savefig("linear-case-a-k.pdf", format='pdf', dpi=1000)
plt.savefig("./figs/sigma.eps", format='eps', dpi=1000)
plt.clf()
