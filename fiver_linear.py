# graphics:
import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *

# scipy:
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, CubicSpline, splrep
from scipy.optimize import minimize, root, root_scalar

# talking to the system
import glob
import re
import os

# parallel support
from mpi4py import MPI
# MPI parameters:
comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()

from cmath import phase

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

import fiver_reader as reader

if(size(sys.argv)>1):
    if csize > 0:
        alpha=double(sys.argv[crank+1])
        print("launched with  alpha = "+str(alpha))
    else:
            alpha=double(sys.argv[1])
else:
    alpha = 0.0

# quantities: Q, Er, Ez, Y, Bz
# parameters: alpha, omega (real), m (whole), Rout = 2.0
chi = alpha * (1.+2.*alpha)/6.
omega = 0.4
m = 1 # not applicable for m=0: the scaling shd then be different
sigma = m-1.
Rin = 0.1
Rout = 1.0
z0 = 10.
npsi = 1000
zmax = 1000.
dzout = 1.e-3
dz0 = 5e-5
Cdiff = 0.5 # multiplier for diffusion-limited time step

r2norm = False # if True, Er/r^2 is averaged in the expr. for B, otherwise Er is averaged and divided by rf^2
abmatch = False # treatment of the inner boundary: if we are using the d/dr(0) = 0 condition;  Unstable, better avoid
shitswitch = True # turns on explicit inner BCs
Ydiffswitch = False # if Ydiff is on, Y is calculated without the EE derivative, and a second-derivative term is added to Ez
# if it is off, we use flux limiter for Ez

# smoothing Y-diffusion parameters
ifAD = False
ADr0 = 2./double(npsi)
ADn = 3.
AD0 = 2.

ifGauss = True

# restart block:
ifrestart = False
restartn = 879
ddir = 'lfiver_alpha0.0_omega0.4'

if ifGauss:
    outdir = 'Glfiver_alpha'+str(alpha)
else:
    outdir = 'lfiver_alpha'+str(alpha)
print(outdir)
os.system('mkdir '+outdir)

def testplot(x, ctr, qua, aqua, qname, q0=None, q1 = None, ztitle=''):
    clf()
    fig =figure()
    plot(x, qua.real, 'k-')
    plot(x, qua.imag, 'k:')
    plot(x, aqua.real, 'r-')
    plot(x, aqua.imag, 'r:')
    if q0 is not None:
        x0 = 2.*x[0]-x[1]
        plot(x0, q0.real, 'or')
        plot(x0, q0.imag, 'xr')
    if q1 is not None:
        x1 = 2.*x[-1]-x[-2]
        plot(x1, q1.real, 'or')
        plot(x1, q1.imag, 'xr')
    plot(x, x*0., 'g--')
    xlabel(r'$\psi$')
    ylabel(r'$q(\psi)$')
    title(ztitle)
    # ylim(-1.,1.)
    # fig.set_size_inches(4.,4.)
    xscale('log')
    # xlim(0.3,0.5)
    fig.tight_layout()
    savefig(outdir+'/lfiver'+qname+'{:05d}.png'.format(ctr))
    close()

def asciiout(fname,s, x, qre, qim, erre, erim, ezre, ezim):
    
    fout = open(fname, 'w')
    
    fout.write('# '+s+'\n')
    
    for k in arange(size(x)):
        fout.write(str(x[k])+' '+str(qre[k])+' '+str(qim[k])+' '+str(erre[k])+' '+str(erim[k])+' '+str(ezre[k])+' '+str(ezim[k])+'\n')
        
    fout.flush()
    fout.close()

def asciiout_f(fname,s, x, bre, bim, yre, yim):
    # TODO: single ASCII output procedure
    fout = open(fname, 'w')
    
    fout.write('# '+s+'\n')
    
    for k in arange(size(x)):
        fout.write(str(x[k])+' '+str(bre[k])+' '+str(bim[k])+' '+str(yre[k])+' '+str(yim[k])+'\n')
        
    fout.flush()
    fout.close()

def asciiread(fname):

    f = open(fname, "r")
    s = f.readline() # header string
    z = double(s[s.find('=')+2:len(s)-1])
    f.close()
    
    # bulk of the data
    lines = loadtxt(fname)
    x = lines[:,0]
    qre = lines[:,1] ; qim = lines[:,2]
    erre = lines[:,3] ; erim = lines[:,4]
    ezre = lines[:,5] ; ezim = lines[:,6]

    return z, x, qre+1.j*qim, erre+1.j*erim, ezre+1.j*ezim

# TODO: check the other inner asymptotics

# Flux limiter:
def FL_phi(r):
    # flux limiter function
    # r is real
    return 2.*r/(1.+r**2)
    
def FL_r(u, u0=None, u1=None):
    # constructing the r function
    FLtol = 1e-8
    rr = zeros(npsi+1, dtype=double)
    # print(size(r.real))
    
    rr[1:-1]= abs(u[2:]-u[1:-1])/abs(u[1:-1]-u[:-2])
    
    if(abs(u[0]-u0) < FLtol * max(abs(u[0]), abs(u0))):
        rr[0] = 0.
    else:
        rr[0] = abs(u[1]-u[0])/abs(u[0]-u0)
        
    if(abs(u[0]-u0) < FLtol * max(abs(u[0]), abs(u0))):
        rr[-1] = 0.
    else:
        rr[-1] = abs(u1-u[-1])/abs(u[-1]-u[-2])

    # rr[0] = 0. ; rr[-1] = 0.

    return rr

def FL_halfstep(u):
    umin = 1.
    rr = zeros(npsi, dtype = double)
    rr = abs(u[1:]-u[:-1])/(abs(u[1:])+abs(u[:-1])+umin)
    return rr

def rfun(z, psi):
    '''
    universal function to compute r(r, psi)
    '''
    # if alpha <= 0.:
    return sqrt(psi) * (z/z0)**alpha * Rout

def rtopsi(r, z):
    return (r/Rout )**2 * (z/z0)**(-2.*alpha) # + log(1.-chi * (r/z)**2)
    
def leftBC(y0, y1):
    '''
    if we want 0 derivative @ half-step, = y0
    if we want 0 derivative in the ghost zone, = 4/3 y0 - 1/3 y1
    if we want 0 derivative in the first cell, =y1
    if we want linear extrapolation, = 2.*y0-y1
    '''
    return  y0 # (y0*2.65-y1)/1.65 # (4.*y0-y1)/3. # (9.*y0-y1)/8.

def abmatchBC(y0, y1, x0, x1, xg):
    acoeff = (y1-y0)/(x1-x0)
    bcoeff = (y0*x1-y1*x0)/(x1-x0)
    return acoeff * xg + bcoeff

def Cmaximum(x,y):
    # chooses the complex number with a larger absolute value
    # ax = abs(x)
    # ay = abs(y)
    
    return maximum(x.real, y.real) + 1.j * maximum(x.imag, y.imag) # x * (ax>ay) + y * (ax<=ay)

def Cminimum(x,y):
    return minimum(x.real, y.real) + 1.j * minimum(x.imag, y.imag)

def byfun(psi, psif, r, rf, Q, Er, Ez, z, Q0=None, Er0=None, Ez0 = None, Q1 = None, Er1 = None, Ez1 = None, adddiff = True):
    '''
    calculates B and Y for visualization purposes and  for the "step"
    '''
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right

    # r = rfun(z, psi)
    # rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)

    if Q0 is None:
        Q0 = leftBC(Q[0], Q[1]) # (3.*Q[0]+Q[1])/4.
    # Er0 = -1.j * (z/z0)**(2.*alpha) * Q0
    if Er0 is None:
        Er0 = -1.j*sqrt(psi0)/r0*Q0
        # Er0 = leftBC(Er[0], Er[1])
    # Ez0 = Ez[0]
    if Ez0 is None:
        Ez0 = leftBC(Ez[0], Ez[1]) # (3.*Ez[0] + Ez[1])/4.
    
    if Q1 is None:
        Q1 = -Q[-1]
    if Er1 is None:
        Er1 = 2.*Er[-1]-Er[-2]
    Er1g = 2.*Er1-Er[-1]

    Ez1 = -Ez[-1] - alpha/z * rf[-1]/sqrt(psif[-1]) * Er1 * 2.

    ee = Ez + alpha*r/z/sqrt(psi) * Er
    ee0 = Ez0 + alpha*r0/z/sqrt(psi0) * Er0
    ee1 = Ez1 + alpha *r1/ z / sqrt(psi1) * Er1g
    # ee1 = -ee[-1]
    
    sigma1 = (sigma+1.)/2.

    Bz_half = zeros(npsi+1, dtype=complex128) # B
        
    Bz_half[1:-1] = 2.j * psif[1:-1]**((1.-sigma)/2.) * ((psi**sigma1*Q)[1:]-(psi**sigma1*Q)[:-1])/dpsi / rf[1:-1]**2/omega
    Bz_half[0] = 2.j * psif[0]**((1.-sigma)/2.) * ((psi**sigma1*Q)[0]-psi0**sigma1*Q0)/dpsi /rf[0]**2/omega
    Bz_half[-1] = 2.j * psif[-1]**((1.-sigma)/2.) * (psi1**sigma1*Q1-(psi**sigma1*Q)[-1])/dpsi/rf[-1]**2/omega
    
    if r2norm:
        Bz_half[1:-1] += m * ((Er/r/sqrt(psi))[1:]+(Er/r/sqrt(psi))[:-1])/2./omega
        Bz_half[0] += (m/2./omega) * (Er[0]/r[0]/(psi[0])+Er0/r0/sqrt(psi0))
        Bz_half[-1] += m * ((Er/r/sqrt(psi[-1]))[-1]+Er1g/r1/sqrt(psi1))/2./omega
    else:
        Bz_half[1:-1] += m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]/sqrt(psif[1:-1])
        Bz_half[0] += (m/2./omega) * (Er[0]+Er0)/rf[0]/sqrt(psif[0])
        Bz_half[-1] += m * Er1/omega /rf[-1]/sqrt(psif[-1])
    

    if shitswitch:
        Bz_half[0] = 1.j /m * (1.-omega) * (ee0+ee[0]) - 0.5j * (Ez0+Ez[0])

    Y_half = zeros(npsi+1, dtype=complex128) # Y

    #diffusive part:
    if adddiff:
        Y_half[1:-1] = (psi**((sigma-1.)/2.)*ee)[1:]-(psi**((sigma-1.)/2.)*ee)[:-1]
        Y_half[0] = (psi**((sigma-1.)/2.)*ee)[0]-psi0**((sigma-1.)/2.)*ee0
        Y_half[-1] = psi1**((sigma-1.)/2.)*ee1-psi[-1]**((sigma-1.)/2.)*ee[-1]
        # rr = FL_r(Y_half[1:-1], u0 = Y_half[0], u1 = Y_half[-1])
        # phi = FL_phi(rr)
        Y_half *= 1.j /omega/rf / dpsi  * psif**((4.-sigma)/2.)
        
        # print("Y = ", Y_half)
        # ii = input("Y")

    # Y part proportional to B
    Y_half += m/2./omega * Bz_half * sqrt(psif)/rf

     # additional terms:
    if r2norm:
        Y_half[1:-1] += 0.25 * alpha / omega / z * ((r*Ez*sqrt(psi))[1:]+(r*Ez*sqrt(psi))[:-1]) - 0.25 * alpha * m / omega / z * ((Q*sqrt(psi)/r)[1:]+(Q*sqrt(psi)/r)[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * ((r*Ez*sqrt(psi))[0]+(r0*Ez0*sqrt(psi0))) - 0.25 * alpha * m / omega / z * ((Q*sqrt(psi)/r)[0]+(Q0*sqrt(psi1)/r0))
        Y_half[-1] += 0.25 * alpha / omega / z * ((r*Ez*sqrt(psi))[-1]+(r1*Ez1*sqrt(psi1))) # - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[-1]+(Q1*exp(psi1/2.)/r1))
    else:
        Y_half[1:-1] += 0.25 * alpha / omega / z * (Ez[1:]+Ez[:-1]) * rf[1:-1]*sqrt(psif[1:-1])  - 0.25 * alpha * m / omega * (sqrt(psif)/ rf)[1:-1] / z * (Q[1:]+Q[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * (Ez[0]+Ez0) * rf[0]*sqrt(psif[0]) - 0.25 * alpha * m / omega  * (sqrt(psif)/ rf)[0] / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * (Ez[-1]+Ez1) * rf[-1]*sqrt(psif[-1])#  - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[-1] / z * (Q[-1]+Q1)

    if shitswitch:
        Y_half[0] = -1.j * sqrt(psif[0])/rf[0] * (ee0+ee[0])/2.

    return Bz_half, Y_half

############ the actual step ############################

def step(psi, psif, Q, Er, Ez, z = 0., Qout = None, Erout = None, Ezout = None, BC_k = None, Q1 = None, Er1 = None, Ez1 = None, Y1 = None, B1 = None, r = None, rf = None):
    '''
    calculates derivatives in z
    '''
    
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right
    
    if r is None:
        r = rfun(z, psi)
    if rf is None:
        rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)
    rf1 = rf[-1]
    
    if Qout is None:
        # Q0 = 0.75* Q[0] + 0.25 * Q[1]
        if abmatch:
            Q0 = abmatchBC(Q[0], Q[1], psi[0]**2, psi[1]**2, psi0**2)
        else:
            # Q0 = 0.75 * Q[0] + 0.25 * Q[1]
            Q0 = leftBC(Q[0], Q[1]) # (4.*Q[0]-Q[1])/3.
    else:
        Q0 = Qout * exp(1.j * BC_k * (z-z0))
    if Erout is None:
        # Er0 = 0.75 * Er[0] + 0.25 * Er[1] # -1.j * (z/z0)**(2.*alpha) * Q0 # Er[0]
        if abmatch:
            Er0 = abmatchBC(Er[0], Er[1], psi[0]**2, psi[1]**2, psi0**2)
            #acoeff = (Er[1]-Er[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            #bcoeff = (Er[0]*exp(2.*psi[1])-Er[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            #Er0 = acoeff * exp(2.*psi0) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Er0 = leftBC(Er[0], Er[1]) # 0.75 * Er[0] + 0.25 * Er[1]
        if shitswitch:
            Er0 = -1.j*sqrt(psi0)/r0*Q0
    else:
        Er0 = Erout * exp(1.j * BC_k * (z-z0))
    if Ezout is None:
        # Ez0 = 0.75 * Ez[0] + 0.25 * Ez[1]
        if abmatch:
            Ez0 = abmatchBC(Ez[0], Ez[1], psi[0]**2, psi[1]**2, psi0**2)
        else:
            Ez0 = leftBC(Ez[0], Ez[1]) # (4.*Ez[0] - Ez[1])/3. # 0.75 * Ez[0] + 0.25 * Ez[1]
            # Ez0 =0.75 * Ez[0] + 0.25 * Ez[1]
    else:
        Ez0 = Ezout * exp(1.j * BC_k * (z-z0))

    # Y0 = Y0 * exp(1.j * BC_k * (z-z0))
    # B0 = B0 * exp(1.j * BC_k * (z-z0))
    
    if Q1 is None:
        Q1 = -Q[-1] # outer ghost zone
    else:
        Q1 *= exp(1.j * BC_k * (z-z0))
    # Q1 = -Q[-1]
        
    Er1g = 2.*Er1-Er[-1]
    '''
    if Er1 is None:
        Er1 = -Er[-1] # 2.*Er[-1]-Er[-2] # no idea what is the correct BC
    else:
        Er1 *= exp(1.j * BC_k * (z-z0))
    '''
    if Ez1 is None:
        Ez1 = -Ez[-1] - alpha/z * rf[-1] /sqrt(psif[-1]) * 2. * Er1
    else:
        Ez1 *= exp(1.j * BC_k * (z-z0))
    
    # Y1 *= exp(1.j * BC_k * z)
    # B1 *= exp(1.j * BC_k * z)
    ee = Ez + alpha*r/z/sqrt(psi) * Er
    ee0 = Ez0 + alpha*r0/z/sqrt(psi0) * Er0
    ee1 = Ez1 + alpha *r1/ z /sqrt(psi1) * Er1g
    
    # print(psi0)
    # ii = input('r0')
    sigma1 = (sigma+1.)/2.
    
    Bz_half, Y_half = byfun(psi, psif, r, rf, Q, Er, Ez, z, Q0 = Q0, Er0 = Er0, Ez0 = Ez0, Q1 = Q1, Er1 = Er1, Ez1 = Ez1, adddiff=not(Ydiffswitch))

    # QQQQQQQQ
    dQ = 1j * (omega+m) * ee - 1.j * chi * omega * r**2/z**2 * Q

    # Er
    
    dEr = alpha/z * Er + (2.+1j * alpha * omega * r**2 /z) * Ez * sqrt(psi)/r + \
    0.5j * ( m * (Bz_half*sqrt(psif)/rf)[1:] + m * (Bz_half*sqrt(psif)/rf)[:-1] - omega * Y_half[1:] - omega * Y_half[:-1]) - 1.j * alpha * m * sqrt(psi) / r / z * Q
    
    # trying to evolve the ghost zone:
    dEr_ghost = 1.j * (m * sqrt(psif)/rf * Bz_half - omega * Y_half)[-1] - alpha / z * (1.+1.j * alpha * omega * rf1**2/z) * Er1
    # Y part:

    # an attempt to rewrite it as a diffusion term without Y
    if Ydiffswitch:
        dEz = zeros(npsi, dtype=complex128)
        dEz[1:-1] += -2.j / omega * psi[1:-1]**(-sigma1) * (psif[2:-1]**3 * ( (psi**((sigma-1.)/2.) * ee)[2:] - (psi**((sigma-1.)/2.) * ee)[1:-1]) - psif[1:-2]**3 * ( (psi**((sigma-1.)/2.) * ee)[1:-1] - (psi**((sigma-1.)/2.) * ee)[:-2])) / (r[1:-1]*dpsi)**2
        dEz[0] += -2.j / omega * psi[0]**(-sigma1) * (psif[1]**3 * ( (psi**((sigma-1.)/2.) * ee)[1] - (psi**((sigma-1.)/2.) * ee)[0]) - psif[0]**3 * ( (psi**((sigma-1.)/2.) * ee)[0] - (psi0**((sigma-1.)/2.) * ee0))) / (r[0]*dpsi)**2
        dEz[-1] = -2.j / omega * psi[-1]**(-sigma1) * (psif[-1]**3 * ( (psi1**((sigma-1.)/2.) * ee1) - (psi**((sigma-1.)/2.) * ee)[-1]) - psif[-2]**3 * ( (psi**((sigma-1.)/2.) * ee)[-1] - (psi**((sigma-1.)/2.) * ee)[-2])) / (r[-1]*dpsi)**2
    else:
        # flux limiter for Ez
        dEz_fluxright = (psif**((sigma+3.)/2.)*Y_half)[1:]
        dEz_fluxleft = (psif**((sigma+3.)/2.)*Y_half)[:-1]
        
        dEz = - 2. * psi**(-(2.+sigma)/2.)/r * (dEz_fluxright - dEz_fluxleft) /dpsi
        # anomalous diffusion:
        if ifAD:
            ADr = FL_halfstep(psif**((sigma+3.)/2.)*Y_half)
            phi = AD0*maximum((ADr-ADr0)/(ADr+ADr0),0.)**ADn*(ADr+ADr0)

            dEz[1:-1] += phi[1:-1] * psi[1:-1]**(-sigma1) * (psif[2:-1]**3 * ( (psi**((sigma-1.)/2.) * ee)[2:] - (psi**((sigma-1.)/2.) * Ez)[1:-1]) - psif[1:-2]**3 * ( (psi**((sigma-1.)/2.) * Ez)[1:-1] - (psi**((sigma-1.)/2.) * Ez)[:-2])) / (dpsi)**2
            dEz[0] += phi[0] * psi[0]**(-sigma1) * (psif[1]**3 * ( (psi**((sigma-1.)/2.) * ee)[1] - (psi**((sigma-1.)/2.) * ee)[0]) - psif[0]**3 * ( (psi**((sigma-1.)/2.) * ee)[0] - (psi0**((sigma-1.)/2.) * ee0))) / (dpsi)**2
            dEz[-1] += phi[-1] * psi[-1]**(-sigma1) * (psif[-1]**3 * ( (psi1**((sigma-1.)/2.) * ee1) - (psi**((sigma-1.)/2.) * ee)[-1]) - psif[-2]**3 * ( (psi**((sigma-1.)/2.) * ee)[-1] - (psi**((sigma-1.)/2.) * ee)[-2])) / (dpsi)**2
        # if (wright.sum() > 0):
        #     dEz[wright] = - 2. * (exp(-psi*(4.+sigma)/2.) * (exp(psif*(sigma+3.)/2.)*Y_half)[1:])[wright]
        # if (wleft.sum() > 0):
        #     dEz[wleft] = 2. * (exp(-psi*(4.+sigma)/2.) * (exp(psif*(sigma+3.)/2.)*Y_half)[:-1])[wleft]
    
    # B part
    dEz +=  - ( (psif**sigma1*Bz_half)[1:] - (psif**sigma1*Bz_half)[:-1]) / psi**sigma1 /r**2 / dpsi

    dEz += - 1.j * alpha * (2.*omega+m) * Er / (r * z * sqrt(psi))- 1.j * (2.*(omega+m)/r**2 + alpha**2 * omega * r**2/z**2) * Ez  - 0.5j * alpha * (2.*omega+m) * (Bz_half[1:]+Bz_half[:-1]) / z + 2.j * chi * (omega+m) * Q/z**2

    aEz =zeros(npsi+1, dtype=complex128) # 2\chi / z^2 * psi^((3-sigma)/2) d_\psi (psi^((sigma-1)/2)er) + 2. * alpha / z psi d_\psi (ez)
    aEz[1:-1] = 2. * chi / z**2 * psif[1:-1]**((2.-sigma)/2.) * ((psi**((sigma-1.)/2.)*Er)[1:]-(psi**((sigma-1.)/2.)*Er)[:-1]) / dpsi +\
        2. * alpha / z * psif[1:-1] ** ((-sigma+1.)/2.) * ((psi**sigma1*Ez)[1:]-(psi**sigma1*Ez)[:-1]) / dpsi
    
    aEz[0] = 2. * chi / z**2 * psif[0]**((2.-sigma)/2.) * ((psi**((sigma-1.)/2.)*Er)[0]-psi0**((sigma-1.)/2.)*Er0) / dpsi +\
        2. * alpha / z * psif[0] ** ((-sigma+1.)/2.) * ((psi**sigma1*Ez)[0]-(psi0**sigma1*Ez0)) / dpsi
#        2. * alpha / z * (Ez[0]-Ez0) / dpsi # zero derivatives?
    aEz[-1] = 2. * chi / z**2 * psif[-1]**((2.-sigma)/2.) * (psi1**((sigma-1.)/2.)*Er1g-(psi**((sigma-1.)/2.)*Er)[-1]) / dpsi +\
        2. * alpha / z * psif[-1] ** ((-sigma+1.)/2.) * ((psi1**sigma1*Ez1)-(psi**sigma1*Ez)[-1]) / dpsi

    #        2. * alpha / z * (Ez1-Ez[-1]) / dpsi # anything better?
    
    dEz += 0.5 * (aEz[1:]+aEz[:-1])

    return dQ, dEr, dEz, dEr_ghost


def onerun(icfile, ifpcolor = False):
    global omega, z0
    # initial conditions involve some value of k and a source file
    # k is read from the file, omega and m do not need to coincide with the global omega and m
   
    if not ifrestart:
        z = z0
        # psi is from Rin/Rout to 1
        psi0 = rtopsi(Rin/Rout, z0) #  2.*log(Rin/Rout)
        psif = zeros(npsi+1)

        psi = (1.-psi0) * (arange(npsi)+0.5)/double(npsi) + psi0
        dpsi = (1.-psi0) / double(npsi)
        
        psif[0] = psi[0] - dpsi/2. ; psif[-1] = psi[-1] + dpsi/2.
        psif[1:-1] = (psi[1:]+psi[:-1])/2.
    
        psi0 = psi[0]-dpsi
        psi1 = psi[-1]+dpsi
    
        r = rfun(z0, psi)
        rf = rfun(z0, psif)
        r0 = rfun(z0, psi[0]-dpsi)
    
        # sigma = sslopefun(omega, m)
        print("sigma = ", sigma)
        print("dpsi = ", dpsi)
        # ii = input('rf')
    
        lines = loadtxt(icfile)
        omega1, m1, R01, kre, kim = lines[0,:] #
    
        if abs(omega-omega1) > 1e-3 * (abs(omega)+abs(omega1)):
            print("omega changed from", omega, "to ", omega1)
            omega = omega1
    
        r1 = lines[1:,0]
        qre1 = lines[1:,1]  ;   qim1 = lines[1:,2]
        yre1 = lines[1:,3]  ;   yim1 = lines[1:,4]

        k = kre + 1.j * kim
        k *= (R01/Rout)**2

        wsafe = abs(m+k*r1**2) > 0.01 # excluding the region near the signular point

        rpole = sqrt(m/abs(k))

        # initial conditions:
        qrefun = interp1d(rtopsi(r1[wsafe]/R01,z0), qre1[wsafe], bounds_error=False, fill_value = 'extrapolate', kind='linear')
        qimfun = interp1d(rtopsi(r1[wsafe]/R01, z0), qim1[wsafe], bounds_error=False, fill_value = 'extrapolate', kind='linear')
        yrefun = interp1d(rtopsi(r1[wsafe]/R01, z0), yre1[wsafe], bounds_error=False, fill_value = 'extrapolate', kind='linear')
        yimfun = interp1d(rtopsi(r1[wsafe]/R01, z0), yim1[wsafe], bounds_error=False, fill_value = 'extrapolate', kind='linear')

        #  Q and Y normalized by r^sigma
        Q = (qrefun(psi) + 1.j * qimfun(psi)) * (Rout/R01)**2
        Y = (yrefun(psi) + 1.j * yimfun(psi))
    
        # now we need the initial conditions for Er and Ez
        Ez = k / (omega+m) * Q # Ez/r^sigma+1
        Bz = (2.j * k /(omega+m) * Q + (2.*omega+m) * Y) / (m + k*r**2) # Bz / r^sigma+1
        Er = m / k * Bz - omega/k * Y - 2.j / (omega+m) * Q # Er/r^sigma

        Qinit = copy(Q)
        Ezinit = copy(Ez)
        Erinit = copy(Er)
        Yinit = copy(Y)
        Yfinit = (yrefun(psif) + 1.j * yimfun(psif))
        Bfinit = (2.j * k /(omega+m) * (qrefun(psif) + 1.j * qimfun(psif)) * (Rout/R01)**2 + (2.*omega+m) * (yrefun(psif) + 1.j * yimfun(psif))) / (m + k*rf**2) # Bz / r^sigma+1

        Q0 = (qrefun(psi0) + 1.j * qimfun(psi0)) * (Rout/R01)**2
        Y0 = (yrefun(psi0) + 1.j * yimfun(psi0))
        Ez0 = k / (omega+m) * Q0
        Bz0 = (2j * k /(omega+m) * Q0 + (2.*omega+m) * Y0) / (m + k*r0**2)
        Er0 = m / k * Bz0 - omega/k * Y0 - 2.j / (omega+m) * Q0 # Er/r^sigma
    
        # psi1 = psi[-1]+dpsi
        psi1f = psif[-1]
        r1f = rfun(z0, psi1f)
        Q1 = (qrefun(psi1f) + 1.j * qimfun(psi1f)) * (Rout/R01)**2
        Y1 = (yrefun(psi1f) + 1.j * yimfun(psi1f))
        Ez1 = k / (omega+m) * Q1
        Bz1 = (2.j * k /(omega+m) * Q1 + (2.*omega+m) * Y1) / (m + k*r1f**2)
        Er1 = m / k * Bz1 - omega/k * Y1 - 2.j / (omega+m) * Q1 # Er/r^sigma
        ctr = 0
        
        if alpha > 0.1:
            # Er -= - sqrt(psi)/r/alpha * Ez  * (1.-exp(-psi*20.))**4
            # Ez -= alpha * r/sqrt(psi) * Er * (1.-exp(-psi*20.))**6 # correction to match the outer BC
            psicutoff= 0.9
            w = psi > 0.5
            Q[w] *= tanh((1.-psi[w])/(1.-psicutoff))
            Er[w] *= tanh((1.-psi[w])/(1.-psicutoff))
            Ez[w] *= tanh((1.-psi[w])/(1.-psicutoff))
    else:
        # restart mode
        ctr = restartn
        fname = ddir+'/lfiver{:05d}'.format(restartn)+'.dat'
        fnameBY = ddir+'/lfiverBY{:05d}'.format(restartn)+'.dat'
        z, psi, Q, Er, Ez = reader.asciiread(fname, ifBY = False)
        z, psif, B, Y = reader.asciiread(fnameBY, ifBY = True)
        dpsi = psi[1]-psi[0]
        psi0 = psi[0]-dpsi
        psi1 = psi[-1]+dpsi
        r = rfun(z, psi) ; rf = rfun(z, psif)
        z0 = z
        Er1 = Er[-1]*1.5-Er[-2]*0.5
        # we still need k
        lines = loadtxt(icfile)
        omega1, m1, R01, kre, kim = lines[0,:] #
        k = kre + 1.j * kim
        k *= (R01/Rout)**2
        Qinit = copy(Q)
        Ezinit = copy(Ez)
        Erinit = copy(Er)
        Yfinit = copy(Y)
        Bfinit = copy(B)

    #
    zstore = z0
    nz = int(round(log(zmax/z0)/log(1.+dzout)))
    
    # two-dimensional plot
    if (ifpcolor):
        q2 = zeros([nz, npsi])
        q2a = zeros([nz, npsi])
        q2abs = zeros([nz, npsi])
        zlist  = []

    # print("Er1  = ", Er1)
    # print("psi1  = ", r1)
    # ii =input('Er1')
    csound = 2. * sqrt((omega+m)/omega)
    dz_CFL = dpsi  / csound
    ddiff = dpsi**2/8. * Cdiff
    print("ddiff = ", ddiff)
    # ii = input("D")
    
    if ifGauss:
        Gc = 0.5 ; Gd = 0.1
        Q = exp(-((psi-Gc)/Gd)**2/2.) + 0.j
        Er = -2.j/m * psi**1.5/r * Q * ((sigma+1.)/2./psi - (psi-Gc)/Gd**2)
        Ez *= 0.
        Er1 = 0.
        # Er -= Er[-1]
        # Q -= exp(-((psi1-Gc)/Gd)**2/2.) + 0.j

    while(ctr < nz):
        if Cdiff < 0.:
            dQ, dEr, dEz, dEr_ghost = step(psi, psif, Q, Er, Ez, z = z, Er1 = Er1, r=r, rf=rf) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # test for the dz estimate
            dratQ = abs(dQ).max()/abs(Q)
            dratEr = abs(dEr).max()/abs(Er)
            dratEz = abs(dEz).max()/abs(Ez)
            dz = median(minimum(1./dratQ, minimum(1./dratEr, 1./dratEz))) * dz0
            if abs(dEr_ghost) > 0.:
                dzghost = abs(Er).max()/abs(dEr_ghost) * dz0
                dz = minimum(dz, dzghost)
        else:
            if (alpha>0.):
                ddiff = (dpsi * exp(psi[0]))**2/8. * (z/z0)**(2.*alpha)
            dz = ddiff

        # print(dEz)
        # print(dratQ.min(), dratEr.min(), dratEz.min())
        # ii = input(dz)
        # dz_CFL = 0.25*dr /vez.max()
        # dz = dz0
        # ii = input(dz)
        # print("z = ", z)
        # Y plots
        # inner edge BCs
        # fixing Er or Ez with 87-89

        dQ1, dEr1, dEz1, dEr_ghost1 = step(psi, psif, Q, Er, Ez, z=z, Er1 = Er1, BC_k = k, r=r, rf=rf) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k1 Runge-Kutta
        dQ2, dEr2, dEz2, dEr_ghost2 = step(psi, psif, Q+dQ1*dz/2., Er+dEr1*dz/2., Ez+dEz1*dz/2., z=z+dz/2., Er1 = Er1+dEr_ghost1*dz/2., r=r, rf=rf) #, Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k2 Runge-Kutta
        dQ3, dEr3, dEz3, dEr_ghost3 = step(psi, psif, Q+dQ2*dz/2., Er+dEr2*dz/2., Ez+dEz2*dz/2., z=z+dz/2., Er1 = Er1+dEr_ghost2*dz/2., r=r, rf=rf) #, Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k3 Runge-Kutta
        dQ4, dEr4, dEz4, dEr_ghost4 = step(psi, psif, Q+dQ3*dz, Er+dEr3*dz, Ez+dEz3*dz, z=z+dz, Er1 = Er1+dEr_ghost3*dz, r=r, rf=rf) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k4 Runge-Kutta

        Q  += (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) * dz/6.
        Er += (dEr1 + 2. * dEr2 + 2. * dEr3 + dEr4) * dz/6.
        Ez += (dEz1 + 2. * dEz2 + 2. * dEz3 + dEz4) * dz/6.
        
        Er1 += (dEr_ghost1 + 2. * dEr_ghost2 + 2. * dEr_ghost3 + dEr_ghost4) * dz / 6.
        
        z += dz
              
        if z >= zstore:
        
            if isnan(Q.sum()):
                print("nan values found")
                print("Q = ", Q)
                print("Er = ", Er)
                print("Ez = ", Ez)
                exit(1)
            # Y and B:
            
            print("z = ", z)
            print("dz = ", dz)
            print("Er[ghost] = ", Er1)
            zstore *= (dzout+1.)
            
            testplot(psi, ctr, Q, Qinit*exp(1j*k*(z-z0)), 'Q', ztitle=r'$z = {:5.5f}$'.format(z))# , q0 = Q0*exp(1j*k*(z-z0)), q1 = Q1*exp(1j*k*(z-z0)))
            testplot(psi, ctr, Ez, Ezinit*exp(1j*k*(z-z0)), 'Ez', ztitle=r'$z = {:5.5f}$'.format(z)) # , q0 = Ez0*exp(1j*k*(z-z0)), q1 = Ez1*exp(1j*k*(z-z0)))
            testplot(psi, ctr, Er, Erinit*exp(1j*k*(z-z0)), 'Er', ztitle=r'$z = {:5.5f}$'.format(z), q1 = Er1) #, q0 = Er0*exp(1j*k*(z-z0)), q1 = Er1*exp(1j*k*(z-z0)))
            
            B, Y = byfun(psi, psif, r, rf, Q, Er, Ez, z, Er1 = Er1, adddiff=True)
            
            testplot(psif, ctr, B, Bfinit*exp(1j*k*(z-z0)), 'B', ztitle=r'$z = {:5.5f}$'.format(z))
            testplot(psif, ctr, Y, Yfinit*exp(1j*k*(z-z0)), 'Y', ztitle=r'$z = {:5.5f}$'.format(z))

            fname = outdir+'/pfiver{:05d}'.format(ctr)+'.dat'
            headerstring = 'z = {:10.10f}'.format(z)
            print(headerstring)
            asciiout(fname, headerstring, psi, Q.real, Q.imag, Er.real, Er.imag, Ez.real, Ez.imag)
            # BY output:
            fnameBY = outdir+'/pfiverBY{:05d}'.format(ctr)+'.dat'
            asciiout_f(fnameBY, headerstring, psif, B.real, B.imag, Y.real, Y.imag)

            if ifpcolor:
                zlist.append(z)
                q2[ctr,:] = Q.real
                q2a[ctr,:] = (Qinit * exp(1j*k*(z-z0))).real
                q2abs[ctr,:] = sqrt(Q.real**2+Q.imag**2)
               # qlist.append(Q.real)
            ctr += 1

    if ifpcolor:
        nz = ctr
        
        # q2 = zeros([nz,nr])
        
        # print(qlist[0]-qlist[-1])
        
        zlist = asarray(zlist)
        z2, psi2 = meshgrid(zlist, psi)
        
        #for k in arange(nz):
        #    q2[k,:] = qlist[k]
        
        print(shape(q2), nz)
        
        clf()
        pcolor(psi, zlist, log10(q2abs)) # , vmin = 0.,vmax = q2.max())
        cb = colorbar()
        cb.set_label(r'$\log_{10}|Q|$')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfiver_abs.png')
        clf()
        pcolor(psi, zlist, q2, vmin = q2.min(),vmax = q2.max())
        colorbar()
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfiver.png')
        clf()
        pcolor(psi, zlist, q2a, vmin = q2a.min(),vmax = q2a.max())
        colorbar()
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfivera.png')

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'fiver*.png' -pix_fmt yuv420p -b 4096k fiver.mp4

if(size(sys.argv)>1):
    # if alpha is set, the simulation starts automatically
    onerun('qysol_o0.4_m1.dat', ifpcolor = True)
