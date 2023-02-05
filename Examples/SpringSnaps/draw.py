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

#### Parameters that control the chart creation
skip = 1
##How many samples are skipped when plotting the chains
mcmcSkip = 100
##KDE How may points
kdePoints = 1000

drawData = True
drawModel1 = True
drawModel2 = True
drawModel3 = True
drawModel4 = True
drawModel3badic = True
drawModel4badic = True
drawModel5 = True
drawModel6 = True

if drawData:
    init_plotting(16,22,800)
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
    plt.savefig("./figs/Case00/data.eps", format='eps', dpi=1000)
    plt.clf()


if drawModel1:
    samples = np.loadtxt('./Case00/chains/Model1-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model1-000.dat')

    init_plotting(16,22,800)
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_samples_k1_k2.eps", format='eps', dpi=1000)
    plt.clf()

    ###Plotting marginal pdfs of a and sigma
    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.xlabel("$k_1$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    true_value = 70.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,100.0])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_k1.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 10.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,100.0])
    plt.xlabel("$k_2$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_k2.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,800.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[3,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,4], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 10.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,50.0])
    plt.xlabel("$t_s$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_tsnaps.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[4,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,5], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.5
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,4.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model1_sigma.eps", format='eps', dpi=1000)
    plt.clf()

#############################################
#############################################
if drawModel2:
    samples = np.loadtxt('./Case00/chains/Model2-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model2-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_samples_k_c.eps", format='eps', dpi=1000)
    plt.clf()

    plt.plot(samples[0,::mcmcSkip], samples[2,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_samples_k_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    plt.plot(samples[1,::mcmcSkip], samples[2,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    ###Plotting marginal pdfs of a and sigma
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 70.0+10.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    true_value = 70.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,3.0])
    plt.xlabel("$K$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_k.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,20.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    #true_value = 0.5
    #plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.ylim([0.0,2.0])
    plt.grid(False)
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model2_sigma.eps", format='eps', dpi=1000)
    plt.clf()

#############################################
#############################################

if drawModel3:
    samples = np.loadtxt('./Case00/chains/Model3-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model3-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    ###Plotting marginal pdfs of a and sigma
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.2)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,125.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.5
    #plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)
    plt.ylim([0.0,2.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_sigma.eps", format='eps', dpi=1000)
    plt.clf()

if drawModel3badic:
    samples = np.loadtxt('./Case00/chains/Model3_BADIC-posterior.dat')
    #samplesCJ = np.loadtxt('./Case00/chains/Model3-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_BADIC_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    ###Plotting marginal pdfs of a and sigma
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.1)
    #xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    #plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,10.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_BADIC_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.04)
    #xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    #plt.plot(xc, yc, 'k--')
    true_value = 0.5
    #plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_BADIC_sigma.eps", format='eps', dpi=1000)
    plt.clf()

#############################################
#############################################
if drawModel4:
    samples = np.loadtxt('./Case00/chains/Model4-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model4-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    plt.plot(samples[1,::mcmcSkip], samples[2,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_samples_sigma_gamma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,125.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.04)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.5
    #plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.08)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.grid(False)

    plt.ylim([0.0,3.0])
    plt.xlabel("$\gamma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_gamma.eps", format='eps', dpi=1000)
    plt.clf()

if drawModel4badic:
    samples = np.loadtxt('./Case00/chains/Model4_BADIC-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model4-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_BADIC_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    plt.plot(samples[1,::mcmcSkip], samples[2,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_BADIC_samples_sigma_gamma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,25.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_BADIC_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.04)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    #true_value = 0.5
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_BADIC_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.08)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$\gamma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_BADIC_gamma.eps", format='eps', dpi=1000)
    plt.clf()

########################################################################
#######################################################################
if drawModel5:
    samples = np.loadtxt('./Case00/chains/Model5-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model5-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_samples_k_c.eps", format='eps', dpi=1000)
    plt.clf()

    samples = np.loadtxt('./Case00/chains/Model5-posterior.dat')
    plt.plot(samples[2,::mcmcSkip], samples[3,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_samples_tau_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    samples = np.loadtxt('./Case00/chains/Model5-posterior.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_samples_k_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.4)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 70.0+10.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    true_value = 70.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.xlabel("$K$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_k.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.ylim([0.0,50.0])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.grid(False)

    plt.xlabel("$\tau$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.ylim([0.0,15.0])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_tau.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[3,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,4], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.grid(False)

    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model5_sigma.eps", format='eps', dpi=1000)
    plt.clf()

########################################################################
########################################################################
if drawModel6:
    samples = np.loadtxt('./Case00/chains/Model6-posterior.dat')
    samplesCJ = np.loadtxt('./Case00/chains/Model6-000.dat')
    plt.plot(samples[0,::mcmcSkip], samples[1,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_samples_c_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    plt.plot(samples[1,::mcmcSkip], samples[2,::mcmcSkip], 'k.')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_samples_sigma_gamma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[0,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,1], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.1
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,150.0])
    plt.xlabel("$c$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_c.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[1,:], kdePoints, bw = 0.1)
    xc,yc = get_marginal_pdf(samplesCJ[:,2], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 0.5
    #plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$\sigma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_sigma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[2,:], kdePoints, bw = 0.08)
    xc,yc = get_marginal_pdf(samplesCJ[:,3], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    true_value = 80.0
    plt.vlines(true_value,0,10.0*max(y), linestyles=[(0,(9,3,4,4))], colors='k')
    plt.grid(False)

    plt.ylim([0.0,4.0])
    plt.xlabel("$\gamma$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_gamma.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    x,y = get_marginal_pdf(samples[3,:], kdePoints, bw = 0.08)
    xc,yc = get_marginal_pdf(samplesCJ[:,4], kdePoints, bw = 0.1)
    plt.plot(x, y, 'k')
    plt.plot(xc, yc, 'k--')
    plt.grid(False)

    plt.ylim([0.0,1.0])
    plt.xlabel("$E[K(0)]$")
    plt.ylabel("pdf")
    plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_k0.eps", format='eps', dpi=1000)
    plt.clf()


###
###
### TO DO CHECK VARIANCE, fill color with lower intensity
############################################
############################################
if drawModel3 or drawModel4 or drawModel6 or drawModel3badic or drawModel4badic:
    ##Making a time series of the stiffness
    ##tk = np.linspace(0.0,20.0, 200)
    tk1 = np.linspace(0.0, 10.0, 100)
    tk2 = np.linspace(10.0, 20.0, 100)
    tk = np.append(tk1,tk2)
    effK = np.zeros(200)
    effK[0:100] = 80.0
    effK[100:200] = 70.0

if drawModel3:
###Plotting state estimation results
#return [time,mean,var,particles,weights]
    [t,mean,var,par,w] =  parseXML('./Case00/Model3SS-state-estimation0.xml')

    #pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(t, mean[0,:], 'b')
    plt.plot(t, mean[0,:]+3.0*np.sqrt(var[0,:]), 'b--')
    plt.plot(t, mean[0,:]-3.0*np.sqrt(var[0,:]), 'b--')
    plt.legend(['True','Model 3'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_x_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[1,:], 'k')
    plt.plot(t, mean[1,:], 'b')
    plt.plot(t, mean[1,:]+3.0*np.sqrt(var[4,:]), 'b--')
    plt.plot(t, mean[1,:]-3.0*np.sqrt(var[4,:]), 'b--')
    plt.legend(['True','Model 3'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_v_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    #plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(tk, effK, 'k')
    plt.plot(t, mean[2,:], 'b')
    plt.fill_between(t,mean[2,:]+3.0*np.sqrt(var[8,:]),mean[2,:]-3.0*np.sqrt(var[8,:]), facecolor='lightblue', alpha=0.5 )
    plt.legend(['True','Model 3'], loc='best')
    plt.grid(False)

    plt.xlabel("Time")
    plt.ylabel("$K$")
    #plt.legend(["TMCMC","CJ","True"])
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model3_k_estimate.eps", format='eps', dpi=1000)
    plt.clf()

############################################
############################################
###Plotting state estimation results
#return [time,mean,var,particles,weights]
if drawModel4:
    [t,mean,var,par,w] =  parseXML('./Case00/Model4SS-state-estimation0.xml')

    #pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(t, mean[0,:], 'b')
    plt.plot(t, mean[0,:]+3.0*np.sqrt(var[0,:]), 'b--')
    plt.plot(t, mean[0,:]-3.0*np.sqrt(var[0,:]), 'b--')
    plt.legend(['True','Model 4'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_x_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[1,:], 'k')
    plt.plot(t, mean[1,:], 'b')
    plt.plot(t, mean[1,:]+3.0*np.sqrt(var[4,:]), 'b--')
    plt.plot(t, mean[1,:]-3.0*np.sqrt(var[4,:]), 'b--')
    plt.legend(['True','Model 4'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_v_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    #plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(tk, effK, 'k')
    plt.plot(t, mean[2,:], 'b')
    plt.fill_between(t,mean[2,:]+3.0*np.sqrt(var[8,:]),mean[2,:]-3.0*np.sqrt(var[8,:]), facecolor='lightblue', alpha=0.5 )
    ##plt.plot(t, mean[2,:]+3.0*np.sqrt(var[8,:]), 'b--')
    ##plt.plot(t, mean[2,:]-3.0*np.sqrt(var[8,:]), 'b--')
    plt.legend(['True','Model 4'], loc='best')
    plt.grid(False)

    plt.xlabel("Time")
    plt.ylabel("$K$")
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model4_k_estimate.eps", format='eps', dpi=1000)
    plt.clf()

if drawModel6:
    [t,mean,var,par,w] =  parseXML('./Case00/Model6SS-state-estimation0.xml')

    #pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(t, mean[0,:], 'b')
    plt.plot(t, mean[0,:]+3.0*np.sqrt(var[0,:]), 'b--')
    plt.plot(t, mean[0,:]-3.0*np.sqrt(var[0,:]), 'b--')
    plt.legend(['True','Model 4'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_x_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    plt.plot(true[-1,:], true[1,:], 'k')
    plt.plot(t, mean[1,:], 'b')
    plt.plot(t, mean[1,:]+3.0*np.sqrt(var[4,:]), 'b--')
    plt.plot(t, mean[1,:]-3.0*np.sqrt(var[4,:]), 'b--')
    plt.legend(['True','Model 4'], loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_v_estimate.eps", format='eps', dpi=1000)
    plt.clf()

    init_plotting(16,22,800)
    #plt.plot(true[-1,:], true[0,:], 'k')
    plt.plot(tk, effK, 'k')
    plt.plot(t, mean[2,:], 'b')
    plt.fill_between(t,mean[2,:]+3.0*np.sqrt(var[8,:]),mean[2,:]-3.0*np.sqrt(var[8,:]), facecolor='lightblue', alpha=0.5 )
    ##plt.plot(t, mean[2,:]+3.0*np.sqrt(var[8,:]), 'b--')
    ##plt.plot(t, mean[2,:]-3.0*np.sqrt(var[8,:]), 'b--')
    plt.legend(['True','Model 3'], loc='best')
    plt.grid(False)

    plt.xlabel("Time")
    plt.ylabel("$K$")
    plt.tight_layout()
    plt.savefig("./figs/Case00/Model6_k_estimate.eps", format='eps', dpi=1000)
    plt.clf()



    if drawModel3badic:
    ###Plotting state estimation results
    #return [time,mean,var,particles,weights]
        [t,mean,var,par,w] =  parseXML('./Case00/Model3SS_BADIC-state-estimation0.xml')

        #pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

        init_plotting(16,22,800)
        plt.plot(true[-1,:], true[0,:], 'k')
        plt.plot(t, mean[0,:], 'b')
        plt.plot(t, mean[0,:]+3.0*np.sqrt(var[0,:]), 'b--')
        plt.plot(t, mean[0,:]-3.0*np.sqrt(var[0,:]), 'b--')
        plt.legend(['True','Model 3'], loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model3_BADIC_x_estimate.eps", format='eps', dpi=1000)
        plt.clf()

        init_plotting(16,22,800)
        plt.plot(true[-1,:], true[1,:], 'k')
        plt.plot(t, mean[1,:], 'b')
        plt.plot(t, mean[1,:]+3.0*np.sqrt(var[4,:]), 'b--')
        plt.plot(t, mean[1,:]-3.0*np.sqrt(var[4,:]), 'b--')
        plt.legend(['True','Model 3'], loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model3_BADIC_v_estimate.eps", format='eps', dpi=1000)
        plt.clf()

        init_plotting(16,22,800)
        #plt.plot(true[-1,:], true[0,:], 'k')
        plt.plot(tk, effK, 'k')
        plt.plot(t, mean[2,:], 'b')
        plt.fill_between(t,mean[2,:]+3.0*np.sqrt(var[8,:]),mean[2,:]-3.0*np.sqrt(var[8,:]), facecolor='lightblue', alpha=0.5 )
        plt.legend(['True','Model 3'], loc='best')
        plt.grid(False)

        plt.xlabel("Time")
        plt.ylabel("$K$")
        plt.xlim([0.0,20.0])
        plt.ylim([0.0,90.0])
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model3_BADIC_k_estimate.eps", format='eps', dpi=1000)
        plt.clf()

    ############################################
    ############################################
    ###Plotting state estimation results
    #return [time,mean,var,particles,weights]
    if drawModel4:
        [t,mean,var,par,w] =  parseXML('./Case00/Model4SS_BADIC-state-estimation0.xml')

        #pfenkf = np.loadtxt('./Case01/PFENKF-state-estimation.dat')

        init_plotting(16,22,800)
        plt.plot(true[-1,:], true[0,:], 'k')
        plt.plot(t, mean[0,:], 'b')
        plt.plot(t, mean[0,:]+3.0*np.sqrt(var[0,:]), 'b--')
        plt.plot(t, mean[0,:]-3.0*np.sqrt(var[0,:]), 'b--')
        plt.legend(['True','Model 4'], loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model4_BADIC_x_estimate.eps", format='eps', dpi=1000)
        plt.clf()

        init_plotting(16,22,800)
        plt.plot(true[-1,:], true[1,:], 'k')
        plt.plot(t, mean[1,:], 'b')
        plt.plot(t, mean[1,:]+3.0*np.sqrt(var[4,:]), 'b--')
        plt.plot(t, mean[1,:]-3.0*np.sqrt(var[4,:]), 'b--')
        plt.legend(['True','Model 4'], loc='best')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model4_BADIC_v_estimate.eps", format='eps', dpi=1000)
        plt.clf()

        init_plotting(16,22,800)
        #plt.plot(true[-1,:], true[0,:], 'k')
        plt.plot(tk, effK, 'k')
        plt.plot(t, mean[2,:], 'b')
        plt.fill_between(t,mean[2,:]+3.0*np.sqrt(var[8,:]),mean[2,:]-3.0*np.sqrt(var[8,:]), facecolor='lightblue', alpha=0.5 )
        ##plt.plot(t, mean[2,:]+3.0*np.sqrt(var[8,:]), 'b--')
        ##plt.plot(t, mean[2,:]-3.0*np.sqrt(var[8,:]), 'b--')
        plt.legend(['True','Model 4'], loc='best')
        plt.grid(False)

        plt.xlabel("Time")
        plt.ylabel("$K$")
        plt.tight_layout()
        plt.savefig("./figs/Case00/Model4_BADIC_k_estimate.eps", format='eps', dpi=1000)
        plt.clf()
