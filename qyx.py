import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import minimize, root, root_scalar
import glob
import re
import os

import scipy
# from scipy import scipy.special
from scipy.special import jv, jn_zeros

'''
Real solution for normalized system of equations
'''

from cmath import phase

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]

k = 1. # may be +/- 1

def RHSQQ(r, omega, m):
    krsq  = k * r**2
    return 2. * krsq / (m+krsq) - m
    
def RHSQY(r, omega, m):
    krsq  = k * r**2
    return (m**2 - 2. * omega * krsq)/ (m+krsq)
    
def RHSYQ(r, omega, m):
    krsq = k * r**2
    return (m+krsq)
    
def RHSYY(r, omega, m):
    krsq = k * r**2
    return -(2.*(omega+m) * m + krsq * (2.+m) )/ (2.*(omega+m)+krsq)
    # (2.*m*(omega+m) + (2.*omega+5.*m)*krsq + 3.*krsq**2) / (m+krsq) / (2. * omega + 2. * m + krsq) - sslope

def normcurve(omega, m):

    x0 = 1e-5
    xmax = 100.
    lnxmax = log(xmax)

    lnx = log(x0)
    
    dlnx = 1e-4
    
    rstore = 0.01
    drout = 0.01
    
    rlist = []
    qlist = []
    ylist = []
    
    Q = 1. ; Y = 1.
    
    fout = open('qyx.dat', 'w+')
    
    while (lnx < lnxmax):
        r = exp(lnx)
        dQ = RHSQQ(r, omega, m)*Q + RHSQY(r, omega, m) * Y
        dY = RHSYQ(r, omega, m)*Q + RHSYY(r, omega, m) * Y

        Q1 = Q + dQ*dlnx/2.
        Y1 = Y + dY*dlnx/2.
        lnx1 = lnx + dlnx/2.
        
        r1 = exp(lnx1)
        
        dQ = RHSQQ(r1, omega, m)*Q1 + RHSQY(r1, omega, m) * Y1
        dY = RHSYQ(r1, omega, m)*Q1 + RHSYY(r1, omega, m) * Y1

        Qprev = Q ; Yprev = Y ; rprev = r

        Q += dQ*dlnx
        Y += dY*dlnx
        lnx += dlnx

        if (rprev > (rstore*(1.+drout))):
            r = exp(lnx)
            rlist.append(r)
            qlist.append(Q)
            ylist.append(Y)
            fout.write(str(r)+' '+str(Q)+' '+str(Y)+'\n')
            fout.flush()
            rstore = r
            print(str(r)+' '+str(Q)+' '+str(Y)+'\n')

        # print(exp(lnx), ' ', Q, ' ', Y)
    fout.close()
    
    rlist = asarray(rlist)
    
    jsol = scipy.special.jv(2, sqrt(2.*omega)*rlist) / rlist**(m/2.)
    
    clf()
    fig = figure()
    # plot(rlist, jsol, 'g:')
    plot(rlist, qlist, 'k-')
    plot(rlist, ylist, 'r--')
    xscale('log') # ; yscale('log')
    xlabel(r'$\sqrt{|k|}r$')
    ylabel(r'$\tilde q$, $y$')
    fig.set_size_inches(8.,4.)
    savefig('qyx.png')
    clf()
    fig = figure()
    plot(rlist, jsol, 'g-')
    plot(rlist, qlist, 'k.')
    xscale('log') # ; yscale('log')
    xlabel(r'$\sqrt{|k|}r$')
    ylabel(r'$\tilde q$, $J_2(\sqrt{2\omega k}r)$')
    xlim(10.,xmax)
    ylim(jsol.min(), jsol.max())
    fig.set_size_inches(8.,4.)
    savefig('qyx_J.png')
    
