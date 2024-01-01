import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, CubicSpline, splrep
from scipy.optimize import minimize, root, root_scalar
import glob
import re
import os

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
alpha = 0.
chi = alpha * (1.+2.*alpha)/6.
omega = 0.4
m = 1 # not applicable for m=0: the scaling shd then be different
Rin = 0.1
Rout = 1.0
z0 = 10.
npsi = 1000
zmax = 50.
dzout = 0.001
r2norm = True # if True, Er/r^2 is averaged in the expr. for B, otherwise Er is averaged and divided by rf^2
abmatch = True # treatment of the inner boundary: if we are using the d/dr(0) = 0 condition

def compareplot(file1, file2, ncol = 1):
    
    lines1 = loadtxt(file1)
    x1 = lines1[:,0] ; q1 = lines1[:,ncol]
    lines2 = loadtxt(file2)
    x2 = lines2[:,0] ; q2= lines2[:,ncol]

    clf()
    plot(x1, q1, 'k-')
    plot(x2, q2, 'r:')
    xlabel(r'$\psi$')
    ylabel(r'$Q$')
    savefig('fivercomp.png')
    close()

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
    savefig('fiver'+qname+'{:05d}.png'.format(ctr))
    close()

def asciiout(fname,s, x, qre, qim, erre, erim, ezre, ezim):
    
    fout = open(fname, 'w')
    
    fout.write('# '+s+'\n')
    
    for k in arange(size(x)):
        fout.write(str(x[k])+' '+str(qre[k])+' '+str(qim[k])+' '+str(erre[k])+' '+str(erim[k])+' '+str(ezre[k])+' '+str(ezim[k])+'\n')
        
    fout.flush()
    fout.close()

def rfun(z, psi):
    '''
    universal function to compute r(r, psi)
    '''
    return exp(psi/2.) * (z/z0)**alpha * Rout

def rtopsi(r, z):
    return 2.*log(r/Rout) - 2.*alpha * log(z/z0)

def sslopefun(omega, m):
    return m-1.
    # m/2./(omega+m)*(sqrt(1.+4.*(omega+m)**2+4.*omega*(omega+m)/m**2)-1.)
        # 0.5 * m / (omega + m) * ()
    # (sqrt(((2.*omega+m)/m)**2+(2.*(omega+m))**2)-1.)
    # sqrt(1.0+3.*m*(omega+m)/(omega+2.*m))

sigma = sslopefun(omega, m)

def byfun(psi, psif, Q, Er, Ez, z):
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
    Ez1 = -Ez[-1]
    Er1 = Er[-1]

    ee = Ez + alpha*r/z/exp(-psi/2.) * Er
    ee0 = Ez0 + alpha*r0/z/exp(-psi0/2.) * Er0
    ee1 = Ez1 + alpha *r1/ z / exp(-psi1/2.) * Er1
    ee1 = -ee[-1]

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
    Y_half[1:-1] += 1.j /omega * exp(psif[1:-1]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*ee)[1:]-(exp(psi*(sigma-1.)/2.)*ee)[:-1]) / dpsi
    Y_half[0] += 1.j /omega * exp(psif[0]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*ee)[0]-exp(psi0*(sigma-1.)/2.)*ee0) / dpsi
    Y_half[-1] += 1.j /omega * exp(psif[-1]*(1.-sigma)/2.) * (exp(psi1*(sigma-1.)/2.)*ee1-exp(psi[-1]*(sigma-1.)/2.)* ee[-1]) / dpsi

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
    
    Y_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.

    return Bz_half, Y_half

def step(psi, psif, Q, Er, Ez, sigma, z = 0., Qout = None, Erout = None, Ezout = None, BC_k = None, Q1 = None, Er1 = None, Ez1 = None, Y1 = None, B1 = None):
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
        if (abmatch):
            acoeff = (Q[1]-Q[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Q[0]*exp(2.*psi[1])-Q[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            Q0 = acoeff * exp(2.*psi0) + bcoeff
        else:
            Q0 = 0.75* Q[0] + 0.25 * Q[1]
    else:
        Q0 = Qout * exp(1.j * BC_k * (z-z0))
    if Erout is None:
        if abmatch:
            acoeff = (Er[1]-Er[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Er[0]*exp(2.*psi[1])-Er[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
            Er0 = acoeff * exp(2.*psi0) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Er0 = 0.75 * Er[0] + 0.25 * Er[1]
    else:
        Er0 = Erout * exp(1.j * BC_k * (z-z0))
    if Ezout is None:
        if abmatch:
            acoeff = (Ez[1]-Ez[0])/(exp(2.*psi[1])-exp(2.*psi[0]))
            bcoeff = (Ez[0]*exp(2.*psi[1])-Er[1]*exp(2.*psi[0]))/(exp(2.*psi[1])-exp(2.*psi[0]))
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
        
    if Ez1 is None:
        Ez1 = -Ez[-1] + alpha / z * (Er[-2]-3.*Er[-1]) # using linear extrapolation for Er
    else:
        Ez1 *= exp(1.j * BC_k * (z-z0))
    if Er1 is None:
        Er1 = 2.*Er[-1]-Er[-2] # no idea what is the correct BC
    else:
        Er1 *= exp(1.j * BC_k * (z-z0))
    
    # Y1 *= exp(1.j * BC_k * z)
    # B1 *= exp(1.j * BC_k * z)

    ee = Ez + alpha/z * Er
    ee0 = Ez0 + alpha/z * Er0
    ee1 = Ez1 + alpha / z * Er1
    
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right
    
    r = rfun(z, psi)
    rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)
    
    # print(psi0)
    # ii = input('r0')
    sigma1 = (sigma+1.)/2.
    
    # B_z field:
    Bz_half = zeros(npsi+1, dtype=complex) # B
    Bz_half[1:-1] = 2.j * exp(-sigma1*psif[1:-1]) * ((exp(sigma1*psi)*Q)[1:]-(exp(sigma1*psi)*Q)[:-1])/dpsi/omega /rf[1:-1]**2
    Bz_half[0] = 2.j * exp(-sigma1*psif[0]) * ((exp(sigma1*psi)*Q)[0]-exp(sigma1*psi0)*Q0)/dpsi/omega / rf[0]**2
    Bz_half[-1] = 2.j * exp(-psif[-1]*sigma1) * (exp(sigma1*psi1)*Q1-(exp(sigma1*psi)*Q)[-1])/dpsi/omega/rf[-1]**2
    if r2norm:
        Bz_half[1:-1] += m * ((Er/r**2)[1:]+(Er/r**2)[:-1])/2./omega
        Bz_half[0] += m * (Er[0]/r[0]**2+Er0/r0**2)/2./omega
        Bz_half[-1] += m * ((Er/r**2)[-1]+Er1/r1**2)/2./omega
    else:
        Bz_half[1:-1] += m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]**2
        Bz_half[0] += m * (Er[0]+Er0)/2./omega/rf[0]**2
        Bz_half[-1] += m * (Er[-1]+Er1)/2./omega /rf[-1]**2

    if abmatch:
        acoeff = (Bz_half[2]-Bz_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
        bcoeff = (Bz_half[1]*exp(2.*psif[2])-Bz_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
        Bz_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
    else:
        Bz_half[0] = (2.*Bz_half[1]+Bz_half[2])/3.

    # YYYYYYYYYYYY #
    Y_half = zeros(npsi+1, dtype=complex) # Y

    Y_half = m/2./omega * Bz_half

    # diffusive part:
    #     if (False): #!!!! temporary
    Y_half[1:-1] += 1.j /omega * exp(psif[1:-1]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*ee)[1:]-(exp(psi*(sigma-1.)/2.)*ee)[:-1]) / dpsi
    Y_half[0] += 1.j /omega * exp(psif[0]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*ee)[0]-exp(psi0*(sigma-1.)/2.)*ee0) / dpsi
    Y_half[-1] += 1.j /omega * exp(psif[-1]*(1.-sigma)/2.) * (exp(psi1*(sigma-1.)/2.)*ee1-exp(psi[-1]*(sigma-1.)/2.)* ee[-1]) / dpsi
    #   Y_half[1:-1] += 1.j /omega * (  (ee[1:]-ee[:-1])/dpsi + (sigma-1.)/4. * (ee[1:]+ee[:-1]) )
    #   Y_half[0] += 1.j /omega * (  (ee[0]-ee0)/dpsi + (sigma-1.)/4. * (ee[0]+ee0) )
    #   Y_half[-1] += 1.j /omega * (  (ee1-ee[-1])/dpsi + (sigma-1.)/4. * (ee1+ee[-1]) )
    
    # additional terms:
    if r2norm:
        Y_half[1:-1] += 0.25 * alpha / omega / z * ((r**2*Ez)[1:]+(r**2*Ez)[:-1]) - 0.25 * alpha * m / omega / z * (Q[1:]+Q[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * ((r**2*Ez)[0]+r0**2*Ez0) - 0.25 * alpha * m / omega / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * ((r**2*Ez)[-1]+r1**2*Ez1) - 0.25 * alpha * m / omega / z * (Q[-1]+Q1)
    else:
        Y_half[1:-1] += 0.25 * alpha / omega / z * (Ez[1:]+Ez[:-1]) * rf[1:-1]**2 - 0.25 * alpha * m / omega / z * (Q[1:]+Q[:-1])
        Y_half[0] += 0.25 * alpha / omega / z * (Ez[0]+Ez0) * rf[0]**2 - 0.25 * alpha * m / omega / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * (Ez[-1]+Ez1) * rf[-1]**2 - 0.25 * alpha * m / omega / z * (Q[-1]+Q1)

    if abmatch:
        acoeff = (Y_half[2]-Y_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
        bcoeff = (Y_half[1]*exp(2.*psif[2])-Y_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
        Y_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
    else:
        Y_half[0] = (2.*Y_half[1]+Y_half[2])/3.

    # Q
    dQ = 1.j * (omega+m) * ee - ((sigma+1.) * alpha / z + 1.j * chi * omega * r**2/z**2) * Q

    # Ez
    dEr = alpha/z * (1.-sigma) * Er + (2.+1.j * alpha * omega * r**2 /z) * Ez + 0.5j * (m * Bz_half[1:] + m * Bz_half[:-1] - omega * Y_half[1:] - omega * Y_half[:-1]) - 1.j * alpha * m / z * Q
    
    dEz = - 2. * (exp(-psi*(3.+sigma)/2.) * ((exp(psif*(sigma+3.)/2.)*Y_half)[1:]-(exp(psif*(sigma+3.)/2.)*Y_half)[:-1]) + \
        exp(-psi*sigma1) * ( (exp(psif*sigma1)*Bz_half)[1:] - (exp(psif*sigma1)*Bz_half)[:-1] ) ) / dpsi/r**2 - \
        1.j * alpha * (2.*omega+m) * Er / r / z - 1.j * (2.*(omega+m)/r**2 + alpha**2 * omega * r**2/z**2) * Ez - 0.5j * alpha * (2.*omega+m) * (Bz_half[1:]+Bz_half[:-1]) / z + 2.j * chi * (omega+m) * Q/z**2 # ! the ~ Ez/r^2 term is safe!
    
    aEz =zeros(npsi+1, dtype=complex) # 2\chi / z^2 * psi^((3-sigma)/2) d_\psi (psi^((sigma-1)/2)er) + 2. * alpha / z psi d_\psi (ez)
    aEz[1:-1] = 2. * chi / z**2 * exp(psif[1:-1]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[1:]-(exp(psi*(sigma-1.)/2.)*Er)[:-1]) / dpsi +\
        2. * alpha / z * (Ez[1:]-Ez[:-1]) / dpsi
    
    aEz[0] = 2. * chi / z**2 * exp(psif[0]*(1.-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[0]-exp(psi0*(sigma-1.)/2.)*Er0) / dpsi +\
        2. * alpha / z * (Ez[0]-Ez0) / dpsi # zero derivatives?s
    aEz[-1] = 2. * chi / z**2 * exp(psif[-1]*(1.-sigma)/2.) * (exp(psi1*(sigma-1.)/2.)*Er1-(exp(psi*(sigma-1.)/2.)*Er)[-1]) / dpsi +\
        2. * alpha / z * (Ez1-Ez[-1]) / dpsi # anything better?
    
    dEz += 0.5 * (aEz[1:]+aEz[:-1])

    return dQ, dEr, dEz

def onerun(icfile, iflog = False, ifpcolor = False):
    # initial conditions involve some value of k and a source file
    # k is read from the file, omega and m do not need to coincide with the global omega and m
    
    z = z0
    
    # psi is from Rin/Rout to 1
    
    psi0 = 2.*log(Rin/Rout)
    psi = -psi0 * (arange(npsi)+0.5)/double(npsi) + psi0
    
    # print(psi)
    # ii = input('psi')

    dpsi = -psi0 / double(npsi)

    psif = zeros(npsi+1)
    psif[1:-1] = (psi[1:]+psi[:-1])/2.
    psif[0] = psi[0] - dpsi/2. ; psif[-1] = psi[-1] + dpsi/2.
    
    psi0 = psi[0]-dpsi ; psi1 = psi[-1]+dpsi
    
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
    qrefun = interp1d(2.*log(r1/R01), qre1, bounds_error=False, fill_value = 'extrapolate', kind='cubic')
    qimfun = interp1d(2.*log(r1/R01), qim1, bounds_error=False, fill_value = 'extrapolate', kind='cubic')
    yrefun = interp1d(2.*log(r1/R01), yre1, bounds_error=False, fill_value = 'extrapolate', kind='cubic')
    yimfun = interp1d(2.*log(r1/R01), yim1, bounds_error=False, fill_value = 'extrapolate', kind='cubic')

    #  Q and Y normalized by r^sigma
    Q = (qrefun(psi) + 1j * qimfun(psi)) * (Rout/R01)**2
    Y = (yrefun(psi) + 1j * yimfun(psi))

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

    Q0 = (qrefun(psi0) + 1j * qimfun(psi0)) * (Rout/R01)**2
    Y0 = (yrefun(psi0) + 1j * yimfun(psi0))
    Ez0 = k / (omega+m) * Q0
    Bz0 = (2j * k /(omega+m) * Q0 + (2.*omega+m) * Y0) / (m + k*r0**2)
    Er0 = m / k * Bz0 - omega/k * Y0 - 2.j / (omega+m) * Q0 # Er/r^sigma
    
    # psi1 = psi[-1]+dpsi
    r1 = rfun(z0, psi1)
    Q1 = (qrefun(psi1) + 1j * qimfun(psi1)) * (Rout/R01)**2
    Y1 = (yrefun(psi1) + 1j * yimfun(psi1))
    Ez1 = k / (omega+m) * Q1
    Bz1 = (2j * k /(omega+m) * Q1 + (2.*omega+m) * Y1) / (m + k*r1**2)
    Er1 = m / k * Bz1 - omega/k * Y1 - 2.j / (omega+m) * Q1 # Er/r^sigma
    
    #
    dz0 = 1e-4
    zstore = z0
    nz = int(round(log(zmax/z0)/log(1.+dzout)))
    ctr = 0
    
    # two-dimensional plot
    if (ifpcolor):
        q2 = zeros([nz, npsi])
        q2a = zeros([nz, npsi])
        zlist  = []

    while(ctr < nz):
        dQ, dEr, dEz = step(psi, psif, Q, Er, Ez, sigma, z = z) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # test for the dz estimate
        # print(dQ[0], dEr[0], dEz[0])
        # ii = input('Q')
        # solamp = maximum(abs(Ez).max(), abs(Er).max())
        # vez = sqrt(m * (omega+m) / 2. / omega**2 / r)
        dratQ = abs(dQ).max()/abs(Q)
        dratEr = abs(dEr).max()/abs(Er)
        dratEz = abs(dEz).max()/abs(Ez)
        dz = median(minimum(1./dratQ, minimum(1./dratEr, 1./dratEz))) * 1e-4
        # print(dEz)
        # print("z=", z)
        # ii = input(dz)
        # dz_CFL = 0.25*dr /vez.max()
        # dz = dz0
        # ii = input(dz)

        dQ1, dEr1, dEz1 = step(psi, psif, Q, Er, Ez, sigma, z=z) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k1 Runge-Kutta
        dQ2, dEr2, dEz2 = step(psi, psif, Q+dQ1*dz/2., Er+dEr1*dz/2., Ez+dEz1*dz/2., sigma, z=z+dz/2.) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k2 Runge-Kutta
        dQ3, dEr3, dEz3 = step(psi, psif, Q+dQ2*dz/2., Er+dEr2*dz/2., Ez+dEz2*dz/2., sigma, z=z+dz/2.) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k3 Runge-Kutta
        dQ4, dEr4, dEz4 = step(psi, psif, Q+dQ3*dz, Er+dEr3*dz, Ez+dEz3*dz, sigma, z=z+dz) # , Qout = Q0, Erout = Er0, Ezout = Ez0, BC_k=k, Q1 = Q1, Er1 = Er1, Ez1 = Ez1) # k4 Runge-Kutta

        Q  += (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) * dz/6.
        Er += (dEr1 + 2. * dEr2 + 2. * dEr3 + dEr4) * dz/6.
        Ez += (dEz1 + 2. * dEz2 + 2. * dEz3 + dEz4) * dz/6.
        
        z += dz
              
        if z >= zstore:
            # Y and B:
            
            print("z = ", z)
            print("dz = ", dz)
            zstore *= (dzout+1.)
            
            testplot(exp(psi), ctr, Q, Qinit*exp(1j*k*(z-z0)), 'Q', ztitle=r'$z = {:5.5f}$'.format(z)) # , q0 = Q0*exp(1j*k*(z-z0)), q1 = Q1*exp(1j*k*(z-z0)))
            testplot(exp(psi), ctr, Ez, Ezinit*exp(1j*k*(z-z0)), 'Ez', ztitle=r'$z = {:5.5f}$'.format(z)) # , q0 = Ez0*exp(1j*k*(z-z0)), q1 = Ez1*exp(1j*k*(z-z0)))
            testplot(exp(psi), ctr, Er, Erinit*exp(1j*k*(z-z0)), 'Er', ztitle=r'$z = {:5.5f}$'.format(z)) # , q0 = Er0*exp(1j*k*(z-z0)), q1 = Er1*exp(1j*k*(z-z0)))
 
            B, Y = byfun(psi, psif, Q, Er, Ez, z)
            testplot(exp(psif), ctr, B, Bfinit*exp(1j*k*(z-z0)), 'B', ztitle=r'$z = {:5.5f}$'.format(z))
            testplot(exp(psif), ctr, Y, Yfinit*exp(1j*k*(z-z0)), 'Y', ztitle=r'$z = {:5.5f}$'.format(z))

            fname = 'fiver{:05d}'.format(ctr)+'.dat'
            headerstring = 'z = {:10.10f}'.format(z)
            print(headerstring)
            asciiout(fname, headerstring, exp(psi), Q.real, Q.imag, Er.real, Er.imag, Ez.real, Ez.imag)
            
            if ifpcolor:
                zlist.append(z)
                q2[ctr,:] = Q.real
                q2a[ctr,:] = (Qinit * exp(1j*k*(z-z0))).real
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
        pcolor(exp(psi), zlist, q2, vmin = q2a.min(),vmax = q2a.max())
        colorbar()
        contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig('fiver.png')
        clf()
        pcolor(exp(psi), zlist, q2a, vmin = q2a.min(),vmax = q2a.max())
        colorbar()
        contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
        xlabel(r'$\psi$')
        ylabel(r'$z$')
        savefig('fivera.png')

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'fiver*.png' -pix_fmt yuv420p -b 4096k fiver.mp4
