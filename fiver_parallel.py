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

# print("size = ", csize)

from timer import Timer # Joonas's timer

from cmath import phase

# quantities: Q, Er, Ez, Y, Bz
# parameters: alpha, omega (real), m (whole), Rout = 2.0
alpha = 0.0
chi = alpha * (1.+2.*alpha)/6.
omega = 0.4
m = 1 # not applicable for m=0: the scaling shd then be different
sigma = m - 1.
sigma1 = (sigma+1.)/2.
Rin = 0.1
Rout = 1.0
z0 = 10.
dz0 = 1.e-4 # scaling for the time step

npsi = 100
nblocks = csize # number of parallel blocks
first = 0 # first block rank
last = nblocks - 1 # last block rank
left = crank - 1
right = crank + 1
oneblock = int(npsi / nblocks) # number of cells in a block

zmax = 50.
dzout = 1e-4

r2norm = False # if True, Er/r^2 is averaged in the expr. for B, otherwise Er is averaged and divided by rf^2
abmatch = False # treatment of the inner boundary: if we are using the d/dr(0) = 0 condition
shitswitch = True
ttest = True

outdir = 'paralpha'+str(alpha)
print(outdir)
os.system('mkdir '+outdir)

# make Y and B global
Bz_half = zeros(oneblock+1, dtype=complex) # B
Y_half = zeros(oneblock+1, dtype=complex) # B
aEz =zeros(oneblock+1, dtype=complex) # 2\chi / z^2 * psi^((3-sigma)/2) d_\psi (psi^((sigma-1)/2)er) + 2. * alpha / z psi d_\psi (ez)

def rfun(z, psi):
    '''
    universal function to compute r(r, psi)
    '''
    if alpha <= 0.:
        return exp(psi/2.) * (z/z0)**alpha * Rout
    else:
        return z/z0 * sqrt((1.-sqrt(1.-4.*chi*exp(psi)*(z/z0)**(2.*(alpha-1.))))/2./chi) * Rout

def rtopsi(r, z):
    return 2.*log(r/Rout) - 2.*alpha * log(z/z0) + log(1.-chi * (r/z)**2)

def asciiout(fname,s, x, qre, qim, erre, erim, ezre, ezim):
    
    fout = open(fname, 'w')
    
    fout.write('# '+s+'\n')
    
    for k in arange(size(x)):
        fout.write(str(x[k])+' '+str(qre[k])+' '+str(qim[k])+' '+str(erre[k])+' '+str(erim[k])+' '+str(ezre[k])+' '+str(ezim[k])+'\n')
        
    fout.flush()
    fout.close()

def BCsend(leftpack_send, rightpack_send):
    leftpack = None ; rightpack = None
    # left = crank-1 ; right = crank+1
    if crank > first:
        comm.send(leftpack_send, dest = left, tag = crank)
    if crank < last:
        comm.send(rightpack_send, dest = right, tag = crank)
    if crank > first:
        leftpack = comm.recv(source = left, tag = left)
    if crank < last:
        rightpack = comm.recv(source = right, tag = right)
    return leftpack, rightpack

def step(psi, psif, Q, Er, Ez, z = 0., Q0 = None, Er0 = None, Ez0 = None, Q1 = None, Er1 = None, Ez1 = None, r = None, rf = None):
    '''
    calculates derivatives in z
    '''
    global Y_half, Bz_half, aEz
    
    dpsi = psi[1]-psi[0]
    psi0 = psi[0]-dpsi # ghost cell to the left
    psi1 = psi[-1]+dpsi # ghost cell to the right
    
    if r is None:
        r = rfun(z, psi)
    if rf is None:
        rf = rfun(z, psif)
    r0 = rfun(z, psi0)
    r1 = rfun(z, psi1)
    
    if crank==first:
        Q0 = 0.75 * Q[0] + 0.25 * Q[1]
        # Er0 = 0.75 * Er[0] + 0.25 * Er[1]
        Ez0 = 0.75 * Ez[0] + 0.25 * Ez[1]
        Er0 = - 1.j * exp(psi0/2.)/r0 * Q0

    Er1g = 2.*Er1-Er[-1]
    if crank==last:
        Q1 = -Q[-1] # outer ghost zone
        Ez1 = -Ez[-1] - alpha/z * rf[-1] * exp(-psif[-1]/2.) * 2. * Er1g

    ee = Ez + alpha*r/z*exp(-psi/2.) * Er
    ee0 = Ez0 + alpha*r0/z*exp(-psi0/2.) * Er0
    ee1 = Ez1 + alpha *r1/ z * exp(-psi1/2.) * Er1g
    
    # print(psi0)
    # ii = input('r0')
    sigma1 = (sigma+1.)/2.
    
    Bz_half[1:-1] = 2.j * exp(-sigma1*psif[1:-1]) * ((exp(sigma1*psi)*Q)[1:]-(exp(sigma1*psi)*Q)[:-1])/dpsi / rf[1:-1]**2/omega
    # + m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]*exp(-psif[1:-1]/2.)
    # m * ((Er/r**2)[1:]+(Er/r**2)[:-1])/2.)/omega
    # Bz_half[0] = 2.j * exp(-sigma1*psif[0]) * ((exp(sigma1*psi)*Q)[0]-exp(sigma1*psi0)*Q0)/dpsi /rf[0]**2/omega
    #  + m * (Er[0]+Er0)/2./omega /rf[0]*exp(-psif[0]/2.) # + m * (Er[0]/r[0]**2+Er0/r0**2)/2.)/omega
    Bz_half[-1] = 2.j * exp(-psif[-1]*sigma1) * (exp(sigma1*psi1)*Q1-(exp(sigma1*psi)*Q)[-1])/dpsi/rf[-1]**2/omega
    # something to do with the BC.
    # + m * (Er[-1]+Er1)/2./omega /rf[-1] * exp(-psif[-1]/2.) # m * ((Er/r**2)[-1]+Er1/r1**2)/2.)/omega
    if crank == first:
        Bz_half[0] = 1.j /m * (1.-omega) * (ee0+ee[0]) - 0.5j * (Ez0+Ez[0])
    else:
        Bz_half[0] = 2.j * exp(-sigma1*psif[0]) * ((exp(sigma1*psi)*Q)[0]-exp(sigma1*psi0)*Q0)/dpsi /rf[0]**2/omega
    
    if r2norm:
        Bz_half[1:-1] += m * ((Er/r*exp(-psi/2.))[1:]+(Er/r*exp(-psi/2.))[:-1])/2./omega
        if crank > first:
            Bz_half[0] += m * (Er[0]/r[0]*exp(-psi[0]/2.)+Er0/r0*exp(-psi0/2.))/2./omega
        # Bz_half[-1] += m * ((Er/r*exp(-psi[-1]/2.))[-1]+Er1/r1*exp(-psi1/2.))/2./omega
    else:
        Bz_half[1:-1] += m * (Er[1:]+Er[:-1])/2./omega /rf[1:-1]*exp(-psif[1:-1]/2.)
        if crank > first:
            Bz_half[0] += m * (Er[0]+Er0)/2./omega/rf[0]*exp(-psif[0]/2.)
        # Bz_half[-1] += m * (Er[-1]+Er1)/2./omega /rf[-1]*exp(-psif[-1]/2.)
    
    Bz_half[-1] += m * Er1 /omega /rf[-1]*exp(-psif[-1]/2.)
    # Bz_half[0] = (2.*Bz_half[1]+Bz_half[2])/3.
    '''
    if crank==last:
        if abmatch:
            acoeff = (Bz_half[2]-Bz_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
            bcoeff = (Bz_half[1]*exp(2.*psif[2])-Bz_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
            Bz_half[0] = acoeff * exp(2.*psif[0]) + bcoeff # (2.*Bz_half[1]+Bz_half[2])/3.
        else:
            Bz_half[0] = (2.*Bz_half[1]+Bz_half[2])/3.
    '''
    # YYYYYYYYYYYYY #
    # Y_half = zeros(oneblock+1, dtype=complex) # Y

    Y_half = m/2./omega * Bz_half * exp(psif/2.)/rf
    
    #diffusive part:
    Y_half[1:-1] += 1.j /omega * exp(psif[1:-1]*(2.-sigma)/2.)/rf[1:-1] * ((exp(psi*(sigma-1.)/2.)*ee)[1:]-(exp(psi*(sigma-1.)/2.)*ee)[:-1]) / dpsi
    if crank == first:
        Y_half[0] = -0.5j * exp(psif[0]/2.)/rf[0] * (ee0+ee[0])
    else:
        Y_half[0] += 1.j /omega * exp(psif[0]*(2.-sigma)/2.)/rf[0] * ((exp(psi*(sigma-1.)/2.)*ee)[0]-exp(psi0*(sigma-1.)/2.)*ee0) / dpsi
    Y_half[-1] += 1.j /omega * exp(psif[-1]*(2.-sigma)/2.) / rf[-1] * (exp(psi1*(sigma-1.)/2.)*ee1-exp(psi[-1]*(sigma-1.)/2.)* ee[-1]) / dpsi
    
    # additional terms:
    if r2norm:
        Y_half[1:-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[1:]+(r*Ez*exp(psi/2.))[:-1]) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[1:]+(Q*exp(psi/2.)/r)[:-1])
        if crank > first:
            Y_half[0] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[0]+(r0*Ez0*exp(psi0/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[0]+(Q0*exp(psi1/2.)/r0))
        Y_half[-1] += 0.25 * alpha / omega / z * ((r*Ez*exp(psi/2.))[-1]+(r1*Ez1*exp(psi1/2.))) - 0.25 * alpha * m / omega / z * ((Q*exp(psi/2.)/r)[-1]+(Q1*exp(psi1/2.)/r1))
    else:
        Y_half[1:-1] += 0.25 * alpha / omega / z * (Ez[1:]+Ez[:-1]) * rf[1:-1]*exp(psif[1:-1]/2.) - 0.25 * alpha * m / omega * (exp(psif/2.)/ rf)[1:-1] / z * (Q[1:]+Q[:-1])
        if crank > first:
            Y_half[0] += 0.25 * alpha / omega / z * (Ez[0]+Ez0) * rf[0]*exp(psif[0]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[0] / z * (Q[0]+Q0)
        Y_half[-1] += 0.25 * alpha / omega / z * (Ez[-1]+Ez1) * rf[-1]*exp(psif[-1]/2.) - 0.25 * alpha * m / omega  * (exp(psif/2.)/ rf)[-1] / z * (Q[-1]+Q1)
    
    # Y_half[-1] = Y_half[-2] * 2. - Y_half[-3] # !!!should it be this way?
    # Y_half[0] = -0.5j * exp(psif[0]/2.)/rf[0] * (ee0+ee[0])
    # Y_half[0] = (2.*Y_half[1]+Y_half[2])/3.
    '''
    if crank==last:
        if abmatch:
            acoeff = (Y_half[2]-Y_half[1])/(exp(2.*psif[2])-exp(2.*psif[1]))
            bcoeff = (Y_half[1]*exp(2.*psif[2])-Y_half[2]*exp(2.*psif[1]))/(exp(2.*psif[2])-exp(2.*psif[1]))
            Y_half[0] = acoeff * exp(2.*psif[0]) + bcoeff
        else:
            Y_half[0] = (2.*Y_half[1]+Y_half[2])/3.
    '''
    
    # QQQQQQQQ
    dQ = 1j * (omega+m) * ee - 1.j * chi * omega * r**2/z**2 * Q

    # Er
    
    dEr = alpha/z * Er + (2.+1j * alpha * omega * r**2 /z) * Ez * exp(psi/2.)/r + 0.5j * (m * (Bz_half*exp(psif/2.)/rf)[1:] + m * (Bz_half*exp(psif/2.)/rf)[:-1] - omega * Y_half[1:] - omega * Y_half[:-1]) - 1.j * alpha * m * exp(psi/2.) / r / z * Q
    
    # trying to evolve the ghost zone:
    dEr_ghost = 1.j * (exp(psif/2.)/rf * Bz_half - omega * Y_half)[-1] # in phif[-1]
    # dEr_ghost = 2.j * (exp(psif/2.)/rf * Bz_half - omega * Y_half)[-1]-dEr[-1] #!!!
    
    # Ez
    dEz = - 2. * (exp(-psi*(4.+sigma)/2.) * ((exp(psif*(sigma+3.)/2.)*Y_half)[1:]-(exp(psif*(sigma+3.)/2.)*Y_half)[:-1]) + \
        exp(-psi*sigma1)/r * ( (exp(psif*sigma1)*Bz_half)[1:] - (exp(psif*sigma1)*Bz_half)[:-1]) ) / dpsi/r + \
        - 1.j * alpha * (2.*omega+m) * Er / (r * z * exp(psi/2.))- 1.j * (2.*(omega+m)/r**2 + alpha**2 * omega * r**2/z**2) * Ez  - 0.5j * alpha * (2.*omega+m) * (Bz_half[1:]+Bz_half[:-1]) / z + 2.j * chi * (omega+m) * Q/z**2
    
    aEz[1:-1] = 2. * chi / z**2 * exp(psif[1:-1]*(-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[1:]-(exp(psi*(sigma-1.)/2.)*Er)[:-1]) / dpsi +\
        2. * alpha / z * exp(-psif[1:-1] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi)*Ez)[1:]-(exp((sigma+1.)/2.*psi)*Ez)[:-1]) / dpsi
    
    aEz[0] = 2. * chi / z**2 * exp(psif[0]*(-sigma)/2.) * ((exp(psi*(sigma-1.)/2.)*Er)[0]-exp(psi0*(sigma-1.)/2.)*Er0) / dpsi +\
        2. * alpha / z * exp(-psif[0] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi)*Ez)[0]-(exp((sigma+1.)/2.*psi0)*Ez0)) / dpsi
#        2. * alpha / z * (Ez[0]-Ez0) / dpsi # zero derivatives?
    aEz[-1] = 2. * chi / z**2 * exp(psif[-1]*(1.-sigma)/2.) * (exp(psi1*(sigma-1.)/2.)*Er1g-(exp(psi*(sigma-1.)/2.)*Er)[-1]) / dpsi +\
        2. * alpha / z * exp(-psif[-1] * (sigma+1.)/2.) * ((exp((sigma+1.)/2.*psi1)*Ez1)-(exp((sigma+1.)/2.*psi)*Ez)[-1]) / dpsi

    #        2. * alpha / z * (Ez1-Ez[-1]) / dpsi # anything better?
    
    dEz += 0.5 * (aEz[1:]+aEz[:-1])

    return dQ, dEr, dEz, dEr_ghost


def runBlock(icfile):
    '''
    running a single domain during an MPI run
    '''
    z = z0
    # setting geometry:
    
    psi0 = 2.*log(Rin/Rout)
    psi = -psi0 * (arange(npsi)+0.5)/double(npsi) + psi0

    dpsi = -psi0 / double(npsi)

    psif = zeros(npsi+1)
    psif[1:-1] = (psi[1:]+psi[:-1])/2.
    psif[0] = psi[0] - dpsi/2. ; psif[-1] = psi[-1] + dpsi/2.

    # indices for the particular block:
    w = arange(oneblock, dtype=int) + crank * oneblock
    wf = arange(oneblock+1, dtype=int) + crank * oneblock

    psi = psi[w]
    psif = psif[wf]
    
    print("crank = ", crank, ": w = ", w[0], w[-1])
    print("crank = ", crank, ": wf = ", wf[0], wf[-1])
    print("crank = ", crank, ", psi = ", psi.min(), "..", psi.max())
    print("crank = ", crank, "dpsi = ", dpsi)

    psi0 = psi[0]-dpsi
    psi1 = psi[-1]+dpsi
    print("crank = ", crank, ", psi0,1 = ", psi0, "..", psi1)

    # radial coordinates:
    r = rfun(z, psi)
    rf = rfun(z, psif)

    # reading the IC from file
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

    # now we need the initial conditions for Er and Ez
    Ez = k / (omega+m) * copy(Q) # Ez/r^sigma+1
    Bz = (2.j * k /(omega+m) * Q + (2.*omega+m) * Y) / (m + k*r**2) # Bz / r^sigma+1
    Er = m / k * Bz - omega/k * Y - 2.j / (omega+m) * Q # Er/r^sigma
    
    Qinit = copy(Q)
    Ezinit = copy(Ez)
    Erinit = copy(Er)
    Yinit = copy(Y)
    Yfinit = (yrefun(psif) + 1.j * yimfun(psif))
    Bfinit = (2.j * k /(omega+m) * (qrefun(psif) + 1.j * qimfun(psif)) * (Rout/R01)**2 + (2.*omega+m) * (yrefun(psif) + 1.j * yimfun(psif))) / (m + k*rf**2) # Bz / r^sigma+1

    r0 = rfun(z, psi[0]-dpsi)
    Q0 = (qrefun(psi0) + 1.j * qimfun(psi0)) * (Rout/R01)**2
    Y0 = (yrefun(psi0) + 1.j * yimfun(psi0))
    Ez0 = k / (omega+m) * Q0
    Bz0 = (2j * k /(omega+m) * Q0 + (2.*omega+m) * Y0) / (m + k*r0**2)
    Er0 = m / k * Bz0 - omega/k * Y0 - 2.j / (omega+m) * Q0 # Er/r^sigma
    
    # psi1 = psi[-1]+dpsi
    psi1f = psif[-1]
    r1f = rfun(z0, psif[-1])
    Q1 = (qrefun(psi1f) + 1.j * qimfun(psi1f)) * (Rout/R01)**2
    Y1 = (yrefun(psi1f) + 1.j * yimfun(psi1f))
    Ez1 = k / (omega+m) * Q1
    Bz1 = (2.j * k /(omega+m) * Q1 + (2.*omega+m) * Y1) / (m + k*r1f**2)
    Er1 = m / k * Bz1 - omega/k * Y1 - 2.j / (omega+m) * Q1 # Er/r^sigma

    # synchronized output:
    zstore = z0
    nz = int(round(log(zmax/z0)/log(1.+dzout)))
    ctr = 0
    
    Q_left = None ; Er_left = None ; Ez_left = None ; Q_right = None ; Er_right = Er1 ; Ez_right = None

    if crank == first:
        thetimer = Timer(["total", "step", "io"], ["BC", "step", "dt"])

    if crank == first:
        thetimer.start("total")
        thetimer.start("io")

    # topology test:
    if ttest:
        if crank > first:
            comm.send({'data': 'from '+str(crank)+' to '+str(left)}, dest = left, tag = crank)
        if crank < last:
            comm.send({'data': 'from '+str(crank)+' to '+str(right)}, dest = right, tag = crank)
        if crank > first:
            leftdata = comm.recv(source = left, tag = left)
            print("I, "+str(crank)+", received from "+str(left)+": "+leftdata['data'])
        if crank < last:
            rightdata = comm.recv(source = right, tag = right)
            print("I, "+str(crank)+", received from "+str(right)+": "+rightdata['data'])
        print("this was topology test\n")
        # tt = input("t")


    while(ctr < nz):
        # Q_left = None ; Er_left = None ; Ez_left = None
        # Q_right = None ; Er_right = Er1 ; Ez_right = None

        # Exchanging BC:
        if crank == first:
            thetimer.start_comp("BC")
        leftpack_send = {'Q1': Q[0], 'Er1': Er[0], 'Ez1': Ez[0]}
        rightpack_send = {'Q0': Q[-1] , 'Er0': Er[-1], 'Ez0': Ez[-1]}
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send)
        if crank > first:
            Q_left = leftpack['Q0'] ; Er_left = leftpack['Er0']  ;  Ez_left = leftpack['Ez0']
        if crank < last:
            Q_right = rightpack['Q1'] ; Er_right = rightpack['Er1']  ;  Ez_right = rightpack['Ez1']
        else:
            Er_right = Er1
        if crank == first:
            thetimer.stop_comp("BC")
            thetimer.start_comp("step")
        # print(Q_right, Er_right, Ez_right)
        # ii = input('right')
        
        dQ1, dEr1, dEz1, dEr_ghost1 = step(psi, psif, Q, Er, Ez, z = z, Q0 = Q_left, Er0 = Er_left, Ez0 = Ez_left, Q1 = Q_right, Er1 = Er_right, Ez1 = Ez_right, r=r, rf=rf)
        if crank == first:
            thetimer.stop_comp("step")
            thetimer.start_comp("dt")
        dz = dz0 * minimum(minimum(abs(Ez).max(),abs(Q).max()),abs(Er).max()) / (abs(dEz1) + abs(dQ1) + abs(dEr1)).max()
        # minimum((abs(Q)/abs(dQ1)).min(), minimum((abs(Er)/abs(dEr1)).min(), (abs(Ez)/abs(dEz1)).min())) * dz0
        # print('crank = ', crank, ": dz(before minimization) = ", dz)
        dz = comm.allreduce(dz, op=MPI.MIN) # calculates one minimal dt
        # print('crank = ', crank, ": dz(after minimization) = ", dz)
        # dz = 1e-8

        if crank == first:
            thetimer.stop_comp("dt")
            thetimer.start_comp("BC")

        leftpack_send = {'Q1': Q[0] + dQ1[0] * dz/2., 'Er1': Er[0] + dEr1[0] * dz/2., 'Ez1': Ez[0] + dEz1[0] * dz/2.}
        rightpack_send = {'Q0': Q[-1] + dQ1[-1] * dz/2. , 'Er0': Er[-1] + dEr1[-1] * dz/2., 'Ez0': Ez[-1] + dEz1[-1] * dz/2.}
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send)
        if crank > first:
            Q_left = leftpack['Q0'] ; Er_left = leftpack['Er0']  ;  Ez_left = leftpack['Ez0']
        if crank < last:
            Q_right = rightpack['Q1'] ; Er_right = rightpack['Er1']  ;  Ez_right = rightpack['Ez1']
        else:
            Er_right = Er1 + dEr_ghost1 * dz/2.

        if crank == first:
            thetimer.stop_comp("BC")
            thetimer.start_comp("step")

        dQ2, dEr2, dEz2, dEr_ghost2 = step(psi, psif, Q+dQ1*dz/2., Er+dEr1*dz/2., Ez+Ez1*dz/2., z=z+dz/2., Q0 = Q_left, Er0 = Er_left, Ez0 = Ez_left, Q1 = Q_right, Er1 = Er_right, Ez1 = Ez_right, r=r, rf=rf)
        
        if crank == first:
            thetimer.stop_comp("step")
            thetimer.start_comp("BC")
            
        leftpack_send = {'Q1': Q[0] + dQ2[0] * dz/2., 'Er1': Er[0] + dEr2[0] * dz/2., 'Ez1': Ez[0] + dEz2[0] * dz/2.}
        rightpack_send = {'Q0': Q[-1] + dQ2[-1] * dz/2. , 'Er0': Er[-1] + dEr2[-1] * dz/2., 'Ez0': Ez[-1] + dEz2[-1] * dz/2.}
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send)
        if crank > first:
            Q_left = leftpack['Q0'] ; Er_left = leftpack['Er0']  ;  Ez_left = leftpack['Ez0']
        if crank < last:
            Q_right = rightpack['Q1'] ; Er_right = rightpack['Er1']  ;  Ez_right = rightpack['Ez1']
        else:
            Er_right = Er1 + dEr_ghost2 * dz/2.
 
        if crank == first:
            thetimer.stop_comp("BC")
            thetimer.start_comp("step")
            
        dQ3, dEr3, dEz3, dEr_ghost3 = step(psi, psif, Q+dQ2*dz/2., Er+dEr2*dz/2., Ez+dEz2*dz/2., z=z+dz/2., Q0 = Q_left, Er0 = Er_left, Ez0 = Ez_left, Q1 = Q_right, Er1 = Er_right, Ez1 = Ez_right, r=r, rf=rf)

        if crank == first:
            thetimer.stop_comp("step")
            thetimer.start_comp("BC")

        leftpack_send = {'Q1': Q[0] + dQ3[0] * dz, 'Er1': Er[0] + dEr3[0] * dz, 'Ez1': Ez[0] + dEz3[0] * dz}
        rightpack_send = {'Q0': Q[-1] + dQ3[-1] * dz , 'Er0': Er[-1] + dEr3[-1] * dz, 'Ez0': Ez[-1] + dEz3[-1] * dz}
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send)
        if crank > first:
            Q_left = leftpack['Q0'] ; Er_left = leftpack['Er0']  ;  Ez_left = leftpack['Ez0']
        if crank < last:
            Q_right = rightpack['Q1'] ; Er_right = rightpack['Er1']  ;  Ez_right = rightpack['Ez1']
        else:
            Er_right = Er1 + dEr_ghost3 * dz

        if crank == first:
            thetimer.stop_comp("BC")
            thetimer.start_comp("step")

        dQ4, dEr4, dEz4, dEr_ghost4 = step(psi, psif, Q+dQ3*dz, Er+dEr3*dz, Ez+dEz3*dz, z=z+dz, Q0 = Q_left, Er0 = Er_left, Ez0 = Ez_left, Q1 = Q_right, Er1 = Er_right, Ez1 = Ez_right, r=r, rf=rf)

        Q  += (dQ1 + 2. * dQ2 + 2. * dQ3 + dQ4) * dz/6.
        Er += (dEr1 + 2. * dEr2 + 2. * dEr3 + dEr4) * dz/6.
        Ez += (dEz1 + 2. * dEz2 + 2. * dEz3 + dEz4) * dz/6.
        
        if crank == last:
            # updating the right ghost cell for Er
            Er1 += (dEr_ghost1 + 2. * dEr_ghost2 + 2. * dEr_ghost3 + dEr_ghost4) * dz / 6.

        z += dz

        if crank == first:
            thetimer.stop_comp("step")
            thetimer.stop("step")
            # thetimer.start_comp("dt")

        if z >= zstore:
            if crank == first:
                thetimer.start("io")
            print("crank = ", crank, ": z = ", z)
            if crank == first:
                # print("dz0 = ", dz0)
                print("dz = ", dz)
            if crank == last:
                print("Er1 = ", Er1)
            # separate ASCII outputs
            fname = outdir+'/par{:05d}'.format(ctr)+'.{:03d}'.format(crank)+'.dat'
            headerstring = 'z = {:10.10f}'.format(z)
            # print(headerstring)
            asciiout(fname, headerstring, exp(psi), Q.real, Q.imag, Er.real, Er.imag, Ez.real, Ez.imag)
                        
            zstore *= (dzout+1.)
            ctr += 1
            if crank == 0:
                thetimer.stop("io")
                thetimer.stats("step")
                thetimer.stats("io")
                thetimer.comp_stats()
                thetimer.start("step") #refresh lap counter (avoids IO profiling)
                thetimer.purge_comps()

runBlock('qysol_o0.4_m1.dat')
