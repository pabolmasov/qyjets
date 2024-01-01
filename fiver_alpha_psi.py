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

# quantities: Q, Er, Ez, Y, Bz
# parameters: alpha, omega (real), m (whole), Rout = 2.0
alpha = 0.0
chi = alpha * (1.+2.*alpha)/6.
omega = 0.4
m = 1 # not applicable for m=0: the scaling shd then be different
Rin = 0.1
Rout = 1.0
z0 = 10.
npsi = 100
zmax = 100.
dzout = 0.001
dz0 = 1e-4

r2norm = True # if True, Er/r^2 is averaged in the expr. for B, otherwise Er is averaged and divided by rf^2
abmatch = True # treatment of the inner boundary: if we are using the d/dr(0) = 0 condition
shitswitch = True

outdir = 'pfiver_alpha'+str(alpha)
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
    fig.tight_layout()
    savefig(outdir+'/pfiver'+qname+'{:05d}.png'.format(ctr))
    close()

def asciiout(fname,s, x, qre, qim, erre, erim, ezre, ezim):
    
    fout = open(fname, 'w')
    
    fout.write('# '+s+'\n')
    
    for k in arange(size(x)):
        fout.write(str(x[k])+' '+str(qre[k])+' '+str(qim[k])+' '+str(erre[k])+' '+str(erim[k])+' '+str(ezre[k])+' '+str(ezim[k])+'\n')
        
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
    
def growthcurve(ddir = 'pfiver_alpha0.5'):
    
    n1 = 0
    n2 = 16
    
    zs = zeros(n2-n1)
    qmax = zeros(n2-n1)
    ezmax = zeros(n2-n1)
    ermax = zeros(n2-n1)
    
    ery_abs = zeros(n2-n1)
    ery_phase = zeros(n2-n1)

    clf()
    fig = figure()
    for k in arange(n2-n1):
        fname = ddir+'/pfiver{:05d}'.format(k+n1)+'.dat'
        print(fname)
        z, x, q, er, ez = asciiread(fname)
        zs[k] = z ; qmax[k] = abs(q).max() ; ezmax[k] = abs(ez).max() ; ermax[k] = abs(er).max()
        ery_abs[k] = abs(er[0])/abs(q[0])
        ery_phase[k] = angle(er[0])-angle(q[0])
        
        if k%3==0:
            plot(x, abs(q), label=r'$z = {:05f}$'.format(z))
        
    legend()
    xscale('log')
    yscale('log')
    xlabel(r'$\psi$')
    ylabel(r'$|Q|$')
    fig.tight_layout()
    savefig('multiline.png')
    
    clf()
    fig =figure()
    plot(zs, qmax, 'k.', label='$\max|Q|$')
    plot(zs, ermax, 'rx', label='$\max|E_r|$')
    plot(zs, ezmax, 'gd', label='$\max|E_z|$')
    xlabel(r'$z$')
    ylabel(r'$\max |Q|$, $\max |E_r|$, $\max |E_z|$')
    xscale('log')
    yscale('log')
    legend()
    fig.set_size_inches(5.,3.)
    fig.tight_layout()
    savefig(ddir+'/qmax.png')
    clf()
    plot(zs, ery_abs, 'k.')
    xscale('log')
    fig.set_size_inches(5.,3.)
    fig.tight_layout()
    savefig(ddir+'/ery.png')
    clf()
    plot(zs, ery_phase % (pi), 'k.')
    plot(zs, zs*0.+pi/2.)
    xscale('log')
    fig.set_size_inches(5.,3.)
    fig.tight_layout()
    savefig(ddir+'/ery_phase.png')

# TODO: check the other inner asymptotics


def rfun(z, psi):
    '''
    universal function to compute r(r, psi)
    '''
    if alpha <= 0.:
        return exp(psi/2.) * (z/z0)**alpha * Rout
    else:
        return z/z0 * sqrt((1.+sqrt(1.-4.*chi*exp(psi)*(z/z0)**(2.*(alpha-1.))))/2./chi) * Rout

def rtopsi(r, z):
    return 2.*log(r/Rout) - 2.*alpha * log(z/z0) + log(1.-chi * (r/z)**2)

def sslopefun(omega, m):
    return m-1.
    # m/2./(omega+m)*(sqrt(1.+4.*(omega+m)**2+4.*omega*(omega+m)/m**2)-1.)
        # 0.5 * m / (omega + m) * ()
    # (sqrt(((2.*omega+m)/m)**2+(2.*(omega+m))**2)-1.)
    # sqrt(1.0+3.*m*(omega+m)/(omega+2.*m))

sigma = sslopefun(omega, m)

def byfun(psi, psif, Q, Er, Ez, z, Er1 = None):
    '''
    calculates B and Y for visualization purposes (and potentially for the "step")
    '''
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right

    r = rfun(z, psi)
    rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)

    Q0 = Q[0]
    Er0 = Er[0]  # -1.j * (z/z0)**(2.*alpha) * Q0
    Ez0 = Ez[0]
    
    Q1 = -Q[-1]
    if Er1 is None:
        Er1 = 2.*Er[-1]-Er[-2]
    # - Er[-1] + 2. *omega/m * (exp(psif/2.)*rf * Bz_half)[-1] - 4.j/m * (exp(psif*(-sigma)/2.)/rf)[-1] * ((exp(psi*(sigma+1.)/2.)*Q)[-1] - exp(psi1*(sigma+1.)/2.)*Q1) / dpsi # the same as for Bz_half (((
    # 2.*Er[-1]-Er[-2]
    # print("psif[-1] = ", psif[-1])
    Ez1 = -Ez[-1] - alpha/z * rf[-1]*exp(-psif[-1]/2.) * (Er1 + Er[-1])

    ee = Ez + alpha*r/z*exp(-psi/2.) * Er
    ee0 = Ez0 + alpha*r0/z*exp(-psi0/2.) * Er0
    ee1 = Ez1 + alpha *r1/ z * exp(-psi1/2.) * Er1
    # ee1 = -ee[-1]

    sigma1 = (sigma+1.)/2.

    Bz_half = zeros(npsi+1, dtype=complex) # B
    
    Bz_half[1:-1] = 2.j * exp(-sigma1*psif[1:-1]) * ((exp(sigma1*psi)*Q)[1:]-(exp(sigma1*psi)*Q)[:-1])/dpsi / rf[1:-1]**2/omega
    Bz_half[0] = 2.j * exp(-sigma1*psif[0]) * ((exp(sigma1*psi)*Q)[0]-exp(sigma1*psi0)*Q0)/dpsi /rf[0]**2/omega
    Bz_half[-1] = 2.j * exp(-psif[-1]*sigma1) * (exp(sigma1*psi1)*Q1-(exp(sigma1*psi)*Q)[-1])/dpsi/rf[-1]**2/omega
    
    if r2norm:
        Bz_half[1:-1] += m * ((Er/r*exp(-psi/2.))[1:]+(Er/r*exp(-psi/2.))[:-1])/2./omega
        Bz_half[0] += (m/2./omega) * (Er[0]/r[0]*exp(-psi[0]/2.)+Er0/r0*exp(-psi0/2.))
        Bz_half[-1] += m * ((Er/r*exp(-psi[-1]/2.))[-1]+Er1/r1*exp(-psi1/2.))/2./omega
    else:
        Bz_half[1:-1] += m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]*exp(-psif[1:-1]/2.)
        Bz_half[0] += (m/2./omega) * (Er[0]+Er0)/rf[0]*exp(-psif[0]/2.)
        Bz_half[-1] += m * (Er[-1]+Er1)/2./omega /rf[-1]*exp(-psif[-1]/2.)
    
    acoeff = (Bz_half[2]-Bz_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
    bcoeff = (Bz_half[1]*exp(2.*psif[2])-Bz_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
    Bz_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
    
    Y_half = zeros(npsi+1, dtype=complex) # Y

    Y_half = m/2./omega * Bz_half * exp(psif/2.)/rf
    
    #diffusive part:
    Y_half[1:-1] += 1.j /omega * exp(psif[1:-1]*(2.-sigma)/2.)/rf[1:-1] * ((exp(psi*(sigma-1.)/2.)*ee)[1:]-(exp(psi*(sigma-1.)/2.)*ee)[:-1]) / dpsi
    Y_half[0] += 1.j /omega * exp(psif[0]*(2.-sigma)/2.)/rf[0] * ((exp(psi*(sigma-1.)/2.)*ee)[0]-exp(psi0*(sigma-1.)/2.)*ee0) / dpsi
    Y_half[-1] += 1.j /omega * exp(psif[-1]*(2.-sigma)/2.) / rf[-1] * (exp(psi1*(sigma-1.)/2.)*ee1-exp(psi[-1]*(sigma-1.)/2.)* ee[-1]) / dpsi
    # Y_half[-1] += 1.j /omega * exp(psif[-1]*(2.-sigma)/2.) / rf[-1] * ((exp(psi1*(sigma-1.)/2.)+exp(psi[-1]*(sigma-1.)/2.)) / dpsi) * ee1
    
     # additional terms:
    if r2norm:
        Y_half[1:-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[1:]+(r*Ez*exp(psi/2.))[:-1]) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[1:]+(Q*exp(psi/2.)/r)[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[0]+(r0*Ez0*exp(psi0/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[0]+(Q0*exp(psi1/2.)/r0))
        Y_half[-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[-1]+(r1*Ez1*exp(psi1/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[-1]+(Q1*exp(psi1/2.)/r1))
    else:
        Y_half[1:-1] += 0.25 * alpha / omega / z * (Ez[1:]+Ez[:-1]) * rf[1:-1]*exp(psif[1:-1]/2.) - 0.25 * alpha * m / omega * (exp(psif/2.)/ rf)[1:-1] / z * (Q[1:]+Q[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * (Ez[0]+Ez0) * rf[0]*exp(psif[0]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[0] / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * (Ez[-1]+Ez1) * rf[-1]*exp(psif[-1]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[-1] / z * (Q[-1]+Q1)

    # Y_half[0] = Y_half[1]
    # Y_half[0] = (2.*Y_half[1]+Y_half[2])/3. # fitting a parabola to ensure zero derivative at psi-1/2
    acoeff = (Y_half[2]-Y_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
    bcoeff = (Y_half[1]*exp(2.*psif[2])-Y_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
    
    # Y_half[-1] = Y_half[-2] * 2. - Y_half[-3] # !!!should it be this way?

    Y_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.

    # Y_half[-1]= -Y_half[-1] # why>??

    return Bz_half, Y_half

def step(psi, psif, Q, Er, Ez, z = 0., Qout = None, Erout = None, Ezout = None, BC_k = None, Q1 = None, Er1 = None, Ez1 = None, Y1 = None, B1 = None):
    '''
    calculates derivatives in z
    '''
    
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right
    
    r = rfun(z, psi)
    rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)
    
    if Qout is None:
        # Q0 = 0.75* Q[0] + 0.25 * Q[1]
        if (abmatch):
            acoeff = (Q[1]-Q[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Q[0]*exp(2.*psi[1])-Q[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            Q0 = acoeff * exp(2.*psi0) + bcoeff
        else:
            Q0 = 0.75 * Q[0] + 0.25 * Q[1]
    else:
        Q0 = Qout * exp(1.j * BC_k * (z-z0))
    if Erout is None:
        # Er0 = 0.75 * Er[0] + 0.25 * Er[1] # -1.j * (z/z0)**(2.*alpha) * Q0 # Er[0]
        if abmatch:
            acoeff = (Er[1]-Er[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Er[0]*exp(2.*psi[1])-Er[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            Er0 = acoeff * exp(2.*psi0) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Er0 = 0.75 * Er[0] + 0.25 * Er[1]
        if shitswitch:
            Er0 = -1.j*exp(psi0/2.)/r0*Q0
    else:
        Er0 = Erout * exp(1.j * BC_k * (z-z0))
    if Ezout is None:
        # Ez0 = 0.75 * Ez[0] + 0.25 * Ez[1]
        if abmatch:
            acoeff = (Ez[1]-Ez[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Ez[0]*exp(2.*psi[1])-Ez[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            Ez0 = acoeff * exp(2.*psi0) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Ez0 = 0.75 * Ez[0] + 0.25 * Ez[1]
    else:
        Ez0 = Ezout * exp(1.j * BC_k * (z-z0))

    # Y0 = Y0 * exp(1.j * BC_k * (z-z0))
    # B0 = B0 * exp(1.j * BC_k * (z-z0))
    
    if Q1 is None:
        Q1 = -Q[-1] # outer ghost zone
    else:
        Q1 *= exp(1.j * BC_k * (z-z0))
    # Q1 = -Q[-1]
        
    '''
    if Er1 is None:
        Er1 = -Er[-1] # 2.*Er[-1]-Er[-2] # no idea what is the correct BC
    else:
        Er1 *= exp(1.j * BC_k * (z-z0))
    '''
    if Ez1 is None:
        Ez1 = -Ez[-1] - alpha/z * rf[-1] * exp(-psif[-1]/2.) * (Er1+Er[-1])
        # Ez1 = -Ez[-1] # + alpha / z * (Er[-2]-3.*Er[-1]) # using linear extrapolation for Er
    else:
        Ez1 *= exp(1.j * BC_k * (z-z0))
    
    # Y1 *= exp(1.j * BC_k * z)
    # B1 *= exp(1.j * BC_k * z)
    ee = Ez + alpha*r/z*exp(-psi/2.) * Er
    ee0 = Ez0 + alpha*r0/z*exp(-psi0/2.) * Er0
    ee1 = Ez1 + alpha *r1/ z * exp(-psi1/2.) * Er1
    
    # print(psi0)
    # ii = input('r0')
    sigma1 = (sigma+1.)/2.
    
    Bz_half = zeros(npsi+1, dtype=complex) # B
    
    Bz_half[1:-1] = 2.j * exp(-sigma1*psif[1:-1]) * ((exp(sigma1*psi)*Q)[1:]-(exp(sigma1*psi)*Q)[:-1])/dpsi / rf[1:-1]**2/omega
    # + m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]*exp(-psif[1:-1]/2.)
    # m * ((Er/r**2)[1:]+(Er/r**2)[:-1])/2.)/omega
    Bz_half[0] = 2.j * exp(-sigma1*psif[0]) * ((exp(sigma1*psi)*Q)[0]-exp(sigma1*psi0)*Q0)/dpsi /rf[0]**2/omega
    #  + m * (Er[0]+Er0)/2./omega /rf[0]*exp(-psif[0]/2.) # + m * (Er[0]/r[0]**2+Er0/r0**2)/2.)/omega
    Bz_half[-1] = 2.j * exp(-psif[-1]*sigma1) * (exp(sigma1*psi1)*Q1-(exp(sigma1*psi)*Q)[-1])/dpsi/rf[-1]**2/omega
    # something to do with the BC.
    # + m * (Er[-1]+Er1)/2./omega /rf[-1] * exp(-psif[-1]/2.) # m * ((Er/r**2)[-1]+Er1/r1**2)/2.)/omega
    
    if r2norm:
        Bz_half[1:-1] += m * ((Er/r*exp(-psi/2.))[1:]+(Er/r*exp(-psi/2.))[:-1])/2./omega
        Bz_half[0] += m * (Er[0]/r[0]*exp(-psi[0]/2.)+Er0/r0*exp(-psi0/2.))/2./omega
        Bz_half[-1] += m * ((Er/r*exp(-psi[-1]/2.))[-1]+Er1/r1*exp(-psi1/2.))/2./omega
    else:
        Bz_half[1:-1] += m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]*exp(-psif[1:-1]/2.)
        Bz_half[0] += m * (Er[0]+Er0)/2./omega/rf[0]*exp(-psif[0]/2.)
        Bz_half[-1] += m * (Er[-1]+Er1)/2./omega /rf[-1]*exp(-psif[-1]/2.)
    
    # Bz_half[0] = (2.*Bz_half[1]+Bz_half[2])/3.
    if shitswitch:
        Bz_half[0] = 1.j /m * (1.-omega) * (ee0+ee[0]) - 0.5j * (Ez0+Ez[0])
    else:
        if abmatch:
            acoeff = (Bz_half[2]-Bz_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
            bcoeff = (Bz_half[1]*exp(2.*psif[2])-Bz_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
            Bz_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Bz_half[0] = (2.*Bz_half[1]+Bz_half[2])/3.
    # YYYYYYYYYYYYY #
    Y_half = zeros(npsi+1, dtype=complex) # Y

    Y_half = m/2./omega * Bz_half * exp(psif/2.)/rf
    
    #diffusive part:
    Y_half[1:-1] += 1.j /omega * exp(psif[1:-1]*(2.-sigma)/2.)/rf[1:-1] * ((exp(psi*(sigma-1.)/2.)*ee)[1:]-(exp(psi*(sigma-1.)/2.)*ee)[:-1]) / dpsi
    Y_half[0] += 1.j /omega * exp(psif[0]*(2.-sigma)/2.)/rf[0] * ((exp(psi*(sigma-1.)/2.)*ee)[0]-exp(psi0*(sigma-1.)/2.)*ee0) / dpsi
    Y_half[-1] += 1.j /omega * exp(psif[-1]*(2.-sigma)/2.) / rf[-1] * (exp(psi1*(sigma-1.)/2.)*ee1-exp(psi[-1]*(sigma-1.)/2.)* ee[-1]) / dpsi
    
    # additional terms:
    if r2norm:
        Y_half[1:-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[1:]+(r*Ez*exp(psi/2.))[:-1]) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[1:]+(Q*exp(psi/2.)/r)[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[0]+(r0*Ez0*exp(psi0/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[0]+(Q0*exp(psi1/2.)/r0))
        Y_half[-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[-1]+(r1*Ez1*exp(psi1/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[-1]+(Q1*exp(psi1/2.)/r1))
    else:
        Y_half[1:-1] += 0.25 * alpha / omega / z * (Ez[1:]+Ez[:-1]) * rf[1:-1]*exp(psif[1:-1]/2.) - 0.25 * alpha * m / omega * (exp(psif/2.)/ rf)[1:-1] / z * (Q[1:]+Q[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * (Ez[0]+Ez0) * rf[0]*exp(psif[0]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[0] / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * (Ez[-1]+Ez1) * rf[-1]*exp(psif[-1]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[-1] / z * (Q[-1]+Q1)
    
    # Y_half[-1] = Y_half[-2] * 2. - Y_half[-3] # !!!should it be this way?

    # Y_half[0] = (2.*Y_half[1]+Y_half[2])/3.
    if shitswitch:
        Y_half[0] = -0.5j * exp(psif[0]/2.)/rf[0] * (ee0+ee[0])
    else:
        if abmatch:
            acoeff = (Y_half[2]-Y_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
            bcoeff = (Y_half[1]*exp(2.*psif[2])-Y_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
            Y_half[0] = acoeff * exp(2.*psif[0]) + bcoeff
        else:
            Y_half[0] = (2.*Y_half[1]+Y_half[2])/3.
        
    # QQQQQQQQ
    dQ = 1j * (omega+m) * ee - 1.j * chi * omega * r**2/z**2 * Q

    # Er
    
    dEr = alpha/z * Er + (2.+1j * alpha * omega * r**2 /z) * Ez * exp(psi/2.)/r + 0.5j * (m * (Bz_half*exp(psif/2.)/rf)[1:] + m * (Bz_half*exp(psif/2.)/rf)[:-1] - omega * Y_half[1:] - omega * Y_half[:-1]) - 1.j * alpha * m * exp(psi/2.) / r / z * Q
    
    # trying to evolve the ghost zone:
    dEr_ghost = 2.j * (exp(psif/2.)/rf * Bz_half - omega * Y_half)[-1]-dEr[-1] #!!!
    
    # Ez
    dEz = - 2. * (exp(-psi*(4.+sigma)/2.) * ((exp(psif*(sigma+3.)/2.)*Y_half)[1:]-(exp(psif*(sigma+3.)/2.)*Y_half)[:-1]) + \
        exp(-psi*sigma1)/r * ( (exp(psif*sigma1)*Bz_half)[1:] - (exp(psif*sigma1)*Bz_half)[:-1]) ) / dpsi/r + \
        - 1.j * alpha * (2.*omega+m) * Er / (r * z * exp(psi/2.))- 1.j * (2.*(omega+m)/r**2 + alpha**2 * omega * r**2/z**2) * Ez  - 0.5j * alpha * (2.*omega+m) * (Bz_half[1:]+Bz_half[:-1]) / z + 2.j * chi * (omega+m) * Q/z**2
    
    aEz =zeros(npsi+1, dtype=complex) # 2\chi / z^2 * psi^((3-sigma)/2) d_\psi (psi^((sigma-1)/2)er) + 2. * alpha / z psi d_\psi (ez)
    aEz[1:-1] = 2. * chi / z**2 * exp(psif[1:-1]*(-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[1:]-(exp(psi*(sigma-1.)/2.)*Er)[:-1]) / dpsi +\
        2. * alpha / z * exp(-psif[1:-1] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi)*Ez)[1:]-(exp((sigma+1.)/2.*psi)*Ez)[:-1]) / dpsi
    
    aEz[0] = 2. * chi / z**2 * exp(psif[0]*(-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[0]-exp(psi0*(sigma-1.)/2.)*Er0) / dpsi +\
        2. * alpha / z * exp(-psif[0] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi)*Ez)[0]-(exp((sigma+1.)/2.*psi0)*Ez0)) / dpsi
#        2. * alpha / z * (Ez[0]-Ez0) / dpsi # zero derivatives?
    aEz[-1] = 2. * chi / z**2 * exp(psif[-1]*(1.-sigma)/2.) * (exp(psi1*(sigma-1.)/2.)*Er1-(exp(psi*(sigma-1.)/2.)*Er)[-1]) / dpsi +\
        2. * alpha / z * exp(-psif[-1] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi1)*Ez1)-(exp((sigma+1.)/2.*psi)*Ez)[-1]) / dpsi

    #        2. * alpha / z * (Ez1-Ez[-1]) / dpsi # anything better?
    
    dEz += 0.5 * (aEz[1:]+aEz[:-1])

    return dQ, dEr, dEz, dEr_ghost

def onerun(icfile, iflog = False, ifpcolor = False):
    # initial conditions involve some value of k and a source file
    # k is read from the file, omega and m do not need to coincide with the global omega and m
    
    z = z0
    
    # psi is from Rin/Rout to 1
    
    psi0 = 2.*log(Rin/Rout)
    psi = -psi0 * (arange(npsi)+0.5)/double(npsi) + psi0
    
    # print(psi.min(), psi.max())
    # ii = input('psi')

    dpsi = -psi0 / double(npsi)

    psif = zeros(npsi+1)
    psif[1:-1] = (psi[1:]+psi[:-1])/2.
    psif[0] = psi[0] - dpsi/2. ; psif[-1] = psi[-1] + dpsi/2.
    
    # print("psif = ", psif)
    # ii  = input('psi')
    
    psi0 = psi[0]-dpsi
    psi1 = psi[-1]+dpsi
    
    r = rfun(z0, psi)
    rf = rfun(z0, psif)
    r0 = rfun(z0, psi[0]-dpsi)

    # dr = (r[1:]-r[:-1]).min()
    
    # sigma = sslopefun(omega, m)
    print("sigma = ", sigma)
    print("dpsi = ", dpsi)
    # ii = input('rf')
    
    lines = loadtxt(icfile)
    omega1, m1, R01, kre, kim = lines[0,:]
    r1 = lines[1:,0]
    qre1 = lines[1:,1]  ;   qim1 = lines[1:,2]
    yre1 = lines[1:,3]  ;   yim1 = lines[1:,4]

    # initial conditions:
    qrefun = interp1d(2.*log(r1/R01), qre1, bounds_error=False, fill_value = 'extrapolate', kind='linear')
    qimfun = interp1d(2.*log(r1/R01), qim1, bounds_error=False, fill_value = 'extrapolate', kind='linear')
    yrefun = interp1d(2.*log(r1/R01), yre1, bounds_error=False, fill_value = 'extrapolate', kind='linear')
    yimfun = interp1d(2.*log(r1/R01), yim1, bounds_error=False, fill_value = 'extrapolate', kind='linear')

    #  Q and Y normalized by r^sigma
    Q = (qrefun(psi) + 1.j * qimfun(psi)) * (Rout/R01)**2
    Y = (yrefun(psi) + 1.j * yimfun(psi))

    k = kre + 1j * kim

    k *= (R01/Rout)**2

    # R01 -> rf[-1]

    print("k = ", k)

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
    r1 = rfun(z0, psi1)
    Q1 = (qrefun(psi1) + 1.j * qimfun(psi1)) * (Rout/R01)**2
    Y1 = (yrefun(psi1) + 1.j * yimfun(psi1))
    Ez1 = k / (omega+m) * Q1
    Bz1 = (2.j * k /(omega+m) * Q1 + (2.*omega+m) * Y1) / (m + k*r1**2)
    Er1 = m / k * Bz1 - omega/k * Y1 - 2.j / (omega+m) * Q1 # Er/r^sigma
    
    #
    zstore = z0
    nz = int(round(log(zmax/z0)/log(1.+dzout)))
    ctr = 0
    
    # two-dimensional plot
    if (ifpcolor):
        q2 = zeros([nz, npsi])
        q2a = zeros([nz, npsi])
        q2abs = zeros([nz, npsi])
        zlist  = []

    # print("Er1  = ", Er1)
    # print("psi1  = ", r1)
    # ii =input('Er1')

    while(ctr < nz):
        dQ, dEr, dEz, dEr_ghost = step(psi, psif, Q, Er, Ez, z = z, Er1 = Er1) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # test for the dz estimate
        # print(dQ[0], dEr[0], dEz[0])
        # ii = input('Q')
        # solamp = maximum(abs(Ez).max(), abs(Er).max())
        # vez = sqrt(m * (omega+m) / 2. / omega**2 / r)
        dratQ = abs(dQ).max()/abs(Q)
        dratEr = abs(dEr).max()/abs(Er)
        dratEz = abs(dEz).max()/abs(Ez)
        dz = median(minimum(1./dratQ, minimum(1./dratEr, 1./dratEz))) * dz0
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

        dQ1, dEr1, dEz1, dEr_ghost1 = step(psi, psif, Q, Er, Ez, z=z, Er1 = Er1, BC_k = k) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k1 Runge-Kutta
        dQ2, dEr2, dEz2, dEr_ghost2 = step(psi, psif, Q+dQ1*dz/2., Er+dEr1*dz/2., Ez+dEz1*dz/2., z=z+dz/2., Er1 = Er1+dEr_ghost*dz/2.) #, Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k2 Runge-Kutta
        dQ3, dEr3, dEz3, dEr_ghost3 = step(psi, psif, Q+dQ2*dz/2., Er+dEr2*dz/2., Ez+dEz2*dz/2., z=z+dz/2., Er1 = Er1+dEr_ghost2*dz/2.) #, Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k3 Runge-Kutta
        dQ4, dEr4, dEz4, dEr_ghost4 = step(psi, psif, Q+dQ3*dz, Er+dEr3*dz, Ez+dEz3*dz, z=z+dz, Er1 = Er1+dEr_ghost3*dz) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k4 Runge-Kutta

        Q  += (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) * dz/6.
        Er += (dEr1 + 2. * dEr2 + 2. * dEr3 + dEr4) * dz/6.
        Ez += (dEz1 + 2. * dEz2 + 2. * dEz3 + dEz4) * dz/6.
        
        Er1 += (dEr_ghost1 + 2. * dEr_ghost2 + 2. * dEr_ghost3 + dEr_ghost4) * dz / 6.
        
        z += dz
              
        if z >= zstore:
            # Y and B:
            
            print("z = ", z)
            print("dz = ", dz)
            print("Er[ghost] = ", Er1)
            zstore *= (dzout+1.)
            
            testplot(exp(psi), ctr, Q, Qinit*exp(1j*k*(z-z0)), 'Q', ztitle=r'$z = {:5.5f}$'.format(z))# , q0 = Q0*exp(1j*k*(z-z0)), q1 = Q1*exp(1j*k*(z-z0)))
            testplot(exp(psi), ctr, Ez, Ezinit*exp(1j*k*(z-z0)), 'Ez', ztitle=r'$z = {:5.5f}$'.format(z)) # , q0 = Ez0*exp(1j*k*(z-z0)), q1 = Ez1*exp(1j*k*(z-z0)))
            testplot(exp(psi), ctr, Er, Erinit*exp(1j*k*(z-z0)), 'Er', ztitle=r'$z = {:5.5f}$'.format(z), q1 = Er1) #, q0 = Er0*exp(1j*k*(z-z0)), q1 = Er1*exp(1j*k*(z-z0)))
            
            B, Y = byfun(psi, psif, Q, Er, Ez, z, Er1 = Er1)
            
            testplot(exp(psif), ctr, B, Bfinit*exp(1j*k*(z-z0)), 'B', ztitle=r'$z = {:5.5f}$'.format(z))
            testplot(exp(psif), ctr, Y, Yfinit*exp(1j*k*(z-z0)), 'Y', ztitle=r'$z = {:5.5f}$'.format(z))

            fname = outdir+'/pfiver{:05d}'.format(ctr)+'.dat'
            headerstring = 'z = {:10.10f}'.format(z)
            print(headerstring)
            asciiout(fname, headerstring, exp(psi), Q.real, Q.imag, Er.real, Er.imag, Ez.real, Ez.imag)
            
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
        pcolor(exp(psi), zlist, log10(q2abs)) # , vmin = 0.,vmax = q2.max())
        cb = colorbar()
        cb.set_label(r'$\log_{10}|Q|$')
        # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfiver_abs.png')
        clf()
        pcolor(exp(psi), zlist, q2, vmin = q2.min(),vmax = q2.max())
        colorbar()
        # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfiver.png')
        clf()
        pcolor(exp(psi), zlist, q2a, vmin = q2a.min(),vmax = q2a.max())
        colorbar()
        # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig(outdir+'/pfivera.png')

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'fiver*.png' -pix_fmt yuv420p -b 4096k fiver.mp4
