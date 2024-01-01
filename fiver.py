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
# parameters: omega (real), m (whole), Rout = 2.0
omega = 0.4
m = 1
Rin = 0.05
Rout = 1.0
nr = 20
zmax = 1.
dzout = 0.1

notchtest = False # if we want the IC to be a point-like perturbation

def sslopefun(omega, m):
    return m-1.
    # m/2./(omega+m)*(sqrt(1.+4.*(omega+m)**2+4.*omega*(omega+m)/m**2)-1.)
        # 0.5 * m / (omega + m) * ()
    # (sqrt(((2.*omega+m)/m)**2+(2.*(omega+m))**2)-1.)
    # sqrt(1.0+3.*m*(omega+m)/(omega+2.*m))

sigma = sslopefun(omega, m)

def step1(r, rf, Q, Er, Ez, sigma, z = 0., BC_k = None, Q0 = 0., Ez0 = 0., Y0 = 0., Bz0 = 0., Er0 = 0., Er1 = 0., yfirst = False, zerodr = True):
    '''
    this version of step combines derivatives with r
    '''
    if BC_k is not None:
        BCmodifier = exp(1.j * BC_k * z)
        # print("Ez0 = ",Ez0, " = ", Ez[0])
        # ii = input("B")
        # Ez0 = Q0 * BC_k / (omega+m)
    else:
        BCmodifier = 1.j
    
    dr = r[1]-r[0]
    r0 = r[0]-dr
    r1 = r[-1]+dr
    
    '''
    Bz = 1j / omega * diffr(Q*r**(sigma+1), rf, r, BCvalue = BCmodifier * Q0 * r0**(sigma+1.)) / r**(sigma+2.) + m / omega * Er / r**2
    Y = m / 2. / omega * Bz + 0.5j * r**(sigma) * diffr(Ez*r**(1.-sigma), rf, r, BCvalue = Ez0 * BCmodifier*r0**(1.-sigma)) / omega
    ii=input('Y0+Bz0 = '+str((Y0+Bz0) * BCmodifier))
    print('left bound ',(m / omega * Er / r**2)[0])
    print(Y[0], Bz[0], Q[0], Er[0], Ez[0])
    ii=input('Y0+Bz0 = '+str((Y+Bz)[0]))
    '''

    Bz_half = zeros(nr+1, dtype=complex) # B r^{1+sigma}
    Bz_half[1:-1] = 1.j * ((Q*r**(sigma+1))[1:]-(Q*r**(sigma+1))[:-1])/dr / omega / rf[1:-1] + m/omega * ((Er*r**(sigma-1.))[1:]+(Er*r**(sigma-1.))[:-1])/2. # b r^sigma+1
    if zerodr:
        Bz_half[0] = 1.j * (sigma+1.) * Q[0]*rf[0]**(sigma-1.) / omega + m/omega * Er[0] * rf[0]**(sigma-1.)
    else:
        Bz_half[0] = 1.j * ((Q*r**(sigma+1))[0] - BCmodifier * Q0*r0**(sigma+1))/dr / omega / rf[0] + m/omega * ((Er*r**(sigma-1.))[0]+(BCmodifier *Er0*r0**(sigma-1.)))/2.
    # Bz_half[0] = Bz_half[1]
    Bz_half[-1] = 1.j * (0. - Q[-1]*r[-1]**(sigma+1.))/dr / omega / rf[-1] + m/omega * ((Er*r**(sigma-1.))[-1]+(BCmodifier * Er1*r1**(sigma-1.)))/2.

    if yfirst:
        Y_half = zeros(nr+1, dtype=complex)  # Y r^(3+sigma)
        Y_half[1:-1] = 0.5j / omega * ( ((Q*r**(sigma+1.))[1:]-(Q*r**(sigma+1.))[:-1])/omega * rf[1:-1] + ((r**(sigma-1.)*Ez)[1:]-(r**(sigma-1.)*Ez)[:-1])*rf[1:-1]**5) / dr + 0.25*(m/omega)**2 * ((Er*r**(sigma+1.))[1:]+(Er*r**(sigma+1.))[:-1])
        Y_half[0] = 0.5j / omega * ( ((Q*r**(sigma+1.))[0]-(BCmodifier * Q0*r0**(sigma+1.)))/omega * rf[0] + ((r**(sigma-1.)*Ez)[0]-(r0**(sigma-1.)*Ez0*BCmodifier))*rf[0]**5 ) / dr + 0.25*(m/omega)**2 * ((Er*r**(sigma+1.))[0]+(BCmodifier * Er0*r0**(sigma+1.)))
        Y_half[-1] = 0.5j / omega * ( (0. - (Q*r**(sigma+1))[-1])/omega * rf[-1] - (0.-(r**(sigma-1.)*Ez)[1]) * rf[-1]**5) / dr + 0.25*(m/omega)**2 * ((Er*r**(sigma+1.))[-1]+(BCmodifier * Er1*r1**(sigma+1.)))
        # Bz_half = copy(Y_half)
        # Bz_half = 2.*omega/m * y
    else:
        Y_half = copy(Bz_half) # Y r^{3+sigma}
        Y_half = m/2./omega * rf**2 * Bz_half
        Y_half[1:-1] += 0.5j / omega * rf[1:-1]**(5.) * ((r**(sigma-1.)*Ez)[1:]-(r**(sigma-1.)*Ez)[:-1])/dr
        if zerodr:
            Y_half[0] += 0.5j * (sigma-1.) / omega * rf[0]**(3.+sigma) * Ez[0]
        else:
            Y_half[0] += 0.5j / omega * rf[0]**(5.) * ((r**(sigma-1.)*Ez)[0]-(BCmodifier * Ez0 * r0**(sigma-1.)))/dr
        Y_half[-1] += 0.5j / omega * rf[-1]**(5.) * (0. - r**(sigma-1.)*Ez)[-1]/dr

    dQ = 1.j * (omega+m) * Ez
    dEr = .5j * m * ((Bz_half/rf**(1.+sigma))[1:]+(Bz_half/rf**(1.+sigma))[:-1]) - .5j * omega * ((Y_half/rf**(3.+sigma))[1:]+(Y_half/rf**(3.+sigma))[:-1]) + 2. * Ez
    dEz = -((Y_half[1:]-Y_half[:-1]) + (Bz_half[1:]-Bz_half[:-1]) * r**2) / (dr * r**(4.+sigma)) - 2.j * (omega+m) * Ez / r**2
    # dEr[-1] = 0.5j * ((m * Bz_half/rf**(1.+sigma) - omega * Y_half/rf**(3.+sigma))[-1] + (m * Bz_half/rf**(1.+sigma) - omega * Y_half/rf**(3.+sigma))[-2])
    if zerodr:
        dEz[0] += -(3.+sigma) * (Y_half[0]+Y_half[1]+(Bz_half[0]+Bz_half[1])*r[0]**2) / r[0]**(5.+sigma) - 2.j * (omega+m) * (Ez/r**2)[0]
    # 0.5j / (omega+m) * ((Y_half[1]-Y_half[0]) / r[0]**2 + (Bz_half[1]-Bz_half[0])) / dr / r[0]**sigma - Ez[0]#
    
    return dQ, dEr, dEz

def onerun(icfile, iflog = False, ifpcolor = False):
    # initial conditions involve some value of k and a source file
    # k is read from the file, omega and m do not need to coincide with the global omega and m
    
    # r mesh
    if iflog:
        r = (Rout / Rin) ** ((arange(nr)/double(nr))) * Rin # currently, only linear scale works
        ii = input('log not tested!')
    else:
        r = (Rout - Rin) * ((arange(nr)/double(nr))) + Rin
        dr = (Rout - Rin) / double(nr)
        
    rf = zeros(nr+1)
    rf[1:-1] = (r[1:]+r[:-1])/2.
    rf[0] = r[0] - dr/2. ; rf[-1] = r[-1] + dr/2.
    
    # dr = (r[1:]-r[:-1]).min()
    
    sigma = sslopefun(omega, m)
    print("sigma = ", sigma)
    print("dr = ", dr)
    # ii = input('rf')
    
    lines = loadtxt(icfile)
    omega1, m1, R01, kre, kim = lines[0,:]
    r1 = lines[1:,0]
    qre1 = lines[1:,1]  ;   qim1 = lines[1:,2]
    yre1 = lines[1:,3]  ;   yim1 = lines[1:,4]

    sigma = sslopefun(omega, m)

    # initial conditions:
    qrefun = interp1d(r1/R01, qre1, bounds_error=False, fill_value = 'extrapolate')
    qimfun = interp1d(r1/R01, qim1, bounds_error=False, fill_value = 'extrapolate')
    yrefun = interp1d(r1/R01, yre1, bounds_error=False, fill_value = 'extrapolate')
    yimfun = interp1d(r1/R01, yim1, bounds_error=False, fill_value = 'extrapolate')

    #  Q and Y normalized by r^sigma
    Q = (qrefun(r/Rout) + 1j * qimfun(r/Rout)) * (Rout/R01)**2
    Y = (yrefun(r/Rout) + 1j * yimfun(r/Rout))

    if notchtest:
        n0 = nr/2
        Q[arange(nr) != n0] *= 0.
        Y[arange(nr) != n0] *= 0.
        Y *= 0.

    k = kre + 1j * kim

    k *= (R01/Rout)**2

    print("k = ", k)

    # now we need the initial conditions for Er and Ez
    Ez = k / (omega+m) * Q # Ez/r^sigma+1
    Bz = (2.j * k /(omega+m) * Q + (2.*omega+m) * Y) / (m + k*r**2) # Bz / r^sigma+1
    Er = m / k * Bz - omega/k * Y - 2.j / (omega+m) * Q # Er/r^sigma
    
    Qinit = copy(Q)
    Ezinit = copy(Ez)
    Erinit = copy(Er)
    Yinit = copy(Y)
    
    r0 = r[0]-dr
    r1 = r[-1]+dr

    Q0 = (qrefun(r0/Rout) + 1j * qimfun(r0/Rout)) * (Rout/R01)**2
    Y0 = (yrefun(r0/Rout) + 1j * yimfun(r0/Rout))
    Ez0 = k / (omega+m) * Q0
    Bz0 = (2j * k /(omega+m) * Q0 + (2.*omega+m) * Y0) / (m + k*r0**2)
    Er0 = m / k * Bz0 - omega/k * Y0 - 2.j / (omega+m) * Q0 # Er/r^sigma
    
    Q1 = (qrefun(r1/Rout) + 1j * qimfun(r1/Rout)) * (Rout/R01)**2
    Y1 = (yrefun(r1/Rout) + 1j * yimfun(r1/Rout))
    Ez1 = k / (omega+m) * Q1
    Bz1 = (2j * k /(omega+m) * Q1 + (2.*omega+m) * Y1) / (m + k*r1**2)
    Er1 = m / k * Bz1 - omega/k * Y1 - 2.j / (omega+m) * Q1 # Er/r^sigma

    if notchtest:
        Q0 = 0.
        Y0 = 0.
        Ez0 = 0.
        Bz0 = 0.

    # ii = input(Y0)
    # ii = input(Bz0)

    #
    z = 0.
    dz0 = 1e-3
    zstore = 0.
    nz = int(round(zmax/dzout))
    ctr = 0
    
    # two-dimensional plot
    if (ifpcolor):
        q2 = zeros([nz, nr])
        q2a = zeros([nz, nr])
        zlist  = []

    while(ctr < nz):
        dQ, dEr, dEz = step1(r, rf, Q, Er, Ez, sigma, z = z, BC_k = k, Q0 = Q0, Ez0 = Ez0, Y0 = Y0, Bz0 = Bz0, Er0 = Er0, Er1 = Er1) # test for the dz estimate
        # print(dQ[0], dEr[0], dEz[0])
        # ii = input('Q')
        vez = sqrt(m * (omega+m) / 2. / omega**2 / r)
        dratQ = abs(dQ).max()/maximum(abs(Q).max(),1e-8)
        dratEr = abs(dEr).max()/maximum(abs(Er).max(), 1e-8)
        dratEz = abs(dEz).max()/maximum(abs(Ez).max(), 1e-8)
        dz = minimum(1./dratQ, minimum(1./dratEr, 1./dratEz)) * 1e-3
        # ii = input(dz)
        dz_CFL = 0.25*dr /vez.max()
        dz = minimum(minimum(dz_CFL, dz), dz0)
        # ii = input(dz_CFL)

        dQ1, dEr1, dEz1 = step1(r, rf, Q, Er, Ez, sigma, z=z, BC_k = k, Q0 = Q0, Ez0 = Ez0, Y0 = Y0, Bz0 = Bz0, Er0 = Er0, Er1 = Er1) # k1 Runge-Kutta
        dQ2, dEr2, dEz2 = step1(r, rf, Q+dQ1*dz/2., Er+dEr1*dz/2., Ez+dEz1*dz/2., sigma, z=z+dz/2., BC_k = k, Q0 = Q0, Ez0 = Ez0, Y0 = Y0, Bz0 = Bz0, Er0 = Er0, Er1 = Er1) # k2 Runge-Kutta
        dQ3, dEr3, dEz3 = step1(r, rf, Q+dQ2*dz/2., Er+dEr2*dz/2., Ez+dEz2*dz/2., sigma, z=z+dz/2., BC_k = k, Q0 = Q0, Ez0 = Ez0, Y0 = Y0, Bz0 = Bz0, Er0 = Er0, Er1 = Er1) # k3 Runge-Kutta
        dQ4, dEr4, dEz4 = step1(r, rf, Q+dQ3*dz, Er+dEr3*dz, Ez+dEz3*dz, sigma, z=z+dz, BC_k = k, Q0 = Q0, Ez0 = Ez0, Y0 = Y0, Bz0 = Bz0, Er0 = Er0, Er1 = Er1) # k4 Runge-Kutta

        Q  += (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) * dz/6.
        Er += (dEr1 + 2. * dEr2 + 2. * dEr3 + dEr4) * dz/6.
        Ez += (dEz1 + 2. * dEz2 + 2. * dEz3 + dEz4) * dz/6.
        
        # at the same time, Ez = dQ/dz / i / (omega+m). WHat if?
        # Ez = (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) / (6.j * (omega+m))
        ##   Ez *= exp(-2.j*(omega+m)*dz/r**2)
        z += dz
        
        #  Ez[0] = - 1j * (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4)[0] /6. / (omega + m) # !!! temporary
        
        if z >= zstore:
            # Y and B:
            
            print("z = ", z)
            print("dz = ", dz)
            zstore += dzout
            clf()
            fig =figure()
            plot(r, (Qinit*exp(1.j*k*z)).real, 'r-')
            plot(r, (Qinit*exp(1.j*k*z)).imag, 'r:')
            plot([r[0]], [(Q0*exp(1.j*k*z)).real], 'ro')
            plot(r, r*0., 'g--')
            plot(r, Q.real, 'k-')
            plot(r, Q.imag, 'k:')
            xlabel(r'$\Omega r$')
            ylabel(r'$q(r)$')
            title(r'$z = {:5.5f}$'.format(z))
            # ylim(-1.,1.)
            # fig.set_size_inches(4.,4.)
            fig.tight_layout()
            savefig('fiverQ{:05d}.png'.format(ctr))
            clf()
            fig =figure()
            plot(r, (Ezinit*exp(1.j*k*z)).real, 'r-')
            plot(r, (Ezinit*exp(1.j*k*z)).imag, 'r:')
            plot([r[0]], [(Ez0*exp(1.j*k*z)).real], 'ro')
            plot(r, r*0., 'g--')
            plot(r, Ez.real, 'k-')
            plot(r, Ez.imag, 'k:')
            xlabel(r'$\Omega r$')
            ylabel(r'$e_z(r)$')
            title(r'$z = {:5.5f}$'.format(z))
            fig.tight_layout()
            # ylim(-1.,1.)
            # fig.set_size_inches(4.,4.)
            savefig('fiverEz{:05d}.png'.format(ctr))
            clf()
            fig =figure()
            plot([r[0]], [(Er0*exp(1.j*k*z)).real], 'ro')
            plot(r, (Erinit*exp(1.j*k*z)).real, 'r-')
            plot(r, (Erinit*exp(1.j*k*z)).imag, 'r:')
            plot(r, r*0., 'g--')
            plot(r, Er.real, 'k-')
            plot(r, Er.imag, 'k:')
            xlabel(r'$\Omega r$')
            ylabel(r'$e_r(r)$')
            title(r'$z = {:5.5f}$'.format(z))
            # ylim(-1.,1.)
            # fig.set_size_inches(4.,4.)
            fig.tight_layout()
            savefig('fiverEr{:05d}.png'.format(ctr))
            close('all')
            if ifpcolor:
                zlist.append(z)
                q2[ctr,:] = Q.real
                q2a[ctr,:] = (Qinit * exp(1j*k*z)).real
               # qlist.append(Q.real)
            ctr += 1

    if ifpcolor:
        nz = ctr
        
        # q2 = zeros([nz,nr])
        
        # print(qlist[0]-qlist[-1])
        
        #for k in arange(nz):
        #    q2[k,:] = qlist[k]
        
        print(shape(q2), nz)
        
        clf()
        pcolor(r, zlist, q2, vmin = -q2a.max()*2.,vmax = q2a.max()*2.)
        xlabel(r'$r$')
        ylabel(r'$z$')
        savefig('fiver.png')
        clf()
        pcolor(r, zlist, q2a, vmin = -q2a.max()*2.,vmax = q2a.max()*2.)
        xlabel(r'$r$')
        ylabel(r'$z$')
        savefig('fivera.png')

# ffmpeg -f image2 -r 15 -pattern_type glob -i 'fiver*.png' -pix_fmt yuv420p -b 4096k fiver.mp4
