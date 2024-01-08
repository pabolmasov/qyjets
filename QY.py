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

from cmath import phase

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

Y0 = 1.

# R0 = 5.

drout = 1e-3

dr0 = 1e-2
tol = 1e-4
drmin = 1e-4

# omega = 0.5
# m = 1

def sslopefun(omega, m):
    if m==0:
        return 3
    else:
        # print(m/2./(omega+m)*(sqrt(1.+4.*(omega+m)**2+4.*omega*(omega+m)/m**2)-1.))
        return abs(m)-1.
    # 0.5 * m / (omega + m) * (sqrt(((2.*omega+m)/m)**2+(2.*(omega+m))**2)-1.)

def RHSQQ(r, k, omega, m):
    if m==0:
        return 2.
    else:
        sslope = sslopefun(omega, m)
        krsq  = k * r**2
        return 2. * krsq / (m+krsq) - (sslope+1.)
    
def RHSQY(r, k, omega, m):
    if m==0:
        return -2j * omega**2 * r / k
    else:
        krsq  = k * r**2
        return 1.j * (omega+m) / k * (m**2 - 2. * omega * krsq)/ (m+krsq)
    
def RHSYQ(r, k, omega, m):
    if m==0:
        return -1.j * k**2 * r / omega
    else:
        krsq = k * r**2
        return -1.j * k * (m+krsq) / ( omega + m )
    
def RHSYY(r,k, omega, m):
    krsq = k * r**2
    if m==0:
        xx = krsq * 0.5 / omega
        return -(1.+3.*xx)/(1.+xx)
    else:
        sslope = sslopefun(omega, m)
        return -(2.*(omega+m) * (1.+sslope) + krsq * (3.+sslope) )/ (2.*(omega+m)+krsq)
    # (2.*m*(omega+m) + (2.*omega+5.*m)*krsq + 3.*krsq**2) / (m+krsq) / (2. * omega + 2. * m + krsq) - sslope

def curvescompare(file1, file2):
    
    lines = loadtxt(file1)
    omega1, m1, R01, kre1, kim1 = lines[0,:]
    r1 = lines[1:,0]
    qre1 = lines[1:,1]  ;   qim1 = lines[1:,2]
    yre1 = lines[1:,3]  ;   yim1 = lines[1:,4]

    lines = loadtxt(file2)
    omega2, m2, R02, kre2, kim2 = lines[0,:]
    r2 = lines[1:,0]
    qre2 = -lines[1:,1]  ;   qim2 = lines[1:,2]
    yre2 = lines[1:,3]  ;   yim2 = -lines[1:,4]

    clf()
    fig=figure()
    plot(r1/R01, qre1/R01**2, 'k-')
    plot(r1/R01, qim1/R01**2, 'k:')
    plot(r1/R01, yre1, 'k--')
    plot(r1/R01, yim1, 'k-.')
    plot(r2/R02, qre2/R02**2, 'r-')
    plot(r2/R02, qim2/R02**2, 'r:')
    plot(r2/R02, yre2, 'r--')
    plot(r2/R02, yim2, 'r-.')
    xlabel(r'$r/R_{\rm out}$')
    ylabel(r'$q(r)/R_{\rm out}^2$, $y(r)$')
    fig.set_size_inches(4.,4.)
    savefig('QYcompare.png')

    q2refun = interp1d(r2/R02, qre2/R02**2, bounds_error=False)
    q2imfun = interp1d(r2/R02, qim2/R02**2, bounds_error=False)

    print(((qre1/R01**2-q2refun(r1))/(qre1/R01**2+q2refun(r1)/R02**2)).min(),((qre1/R01**2-q2refun(r1))/(qre1/R01**2+q2refun(r1)/R02**2)).max())

    clf()
    plot(r1/R01, (qre1/R01**2-q2refun(r1/R01))/sqrt((qre1/R01**2)**2+(qim1/R01**2)**2+(q2refun(r1/R01)/R02**2)**2+(q2imfun(r1/R01)/R02**2)**2), 'k-')
    plot(r1/R01, (qim1/R01**2-q2imfun(r1/R01))/sqrt((qre1/R01**2)**2+(qim1/R01**2)**2+(q2refun(r1/R01)/R02**2)**2+(q2imfun(r1/R01)/R02**2)**2), 'k,:')
    xlabel(r'$r/R_{\rm out}$')
    ylabel(r'$(q_1(r/R_1)-q_2(r/R_2))/(q_1(r/R_1)+q_2(r/R_2))$')
    savefig('DQY.png')


def ksigns(omega, m, R0):

    k0 = [-0.4737*4.,-0.1345*4.]

    res = root(onecurve, k0, args = (omega, m, 1.))

    onecurve(res.x, omega, m, 1., ifplot=True)
    print(res)
    os.system('cp qysol.dat qysol_un.dat')
    res = root(onecurve, [res.x[0]/R0**2, -res.x[1]/R0**2], args = (omega, m, R0))
    onecurve(res.x, omega, m, R0, ifplot=True)
    os.system('cp qysol.dat qysol_st.dat')

    curvescompare('qysol_st.dat', 'qysol_un.dat')


def onecurve(kvec, omega, m, R0, ifplot = False, Q0 = None):
    
    #if no_kim_neg & (kvec[1]< 0.):
    #    return [100., 100.]

    sslope = sslopefun(omega, m)

    k = kvec[0] + kvec[1] * 1j
        
    if ifplot:
        fout = open('qysol.dat', 'w+')
        fout.write('# first line contains omega, m, R, Re(k), and Im(k)\n')
        fout.write('# then, r, Re(Q), Im(Q), Re(Y), Im(Y)\n')
        fout.write(str(omega)+" "+str(m)+" "+str(R0)+" "+str(kvec[0])+" "+str(kvec[1])+"\n")
        
    rlist = [] ; qlist = [] ; ylist = []
    
    r = 0. ; rstore = 0.
    Y = 1. + 0.j
    if Q0 is not None:
        Q = Q0
    else:
        Q = (m * (omega+m)/ k / (sslope+1.)) * 1j

    # maxQ  = 0.
    
    r = drmin
    dr = drmin
    
    while(r<R0):
        dQ = RHSQQ(r,k, omega, m)*Q + RHSQY(r,k, omega, m) * Y
        dY = RHSYQ(r,k, omega, m)*Q + RHSYY(r,k, omega, m) * Y
        
        # adaptive dr/r
        if m==0:
            dr = minimum(maximum(abs(Q/dQ) * tol,abs(Y/dY)*tol),dr0)
        else:
            if abs(Q) > 1e-8:
                dr = minimum(minimum(abs(Q/dQ) * tol,abs(Y/dY)*tol),dr0)
            else:
                dr = minimum(abs(Y/dY)*tol,dr0)
        # dr = minimum(maximum(drmin, drmin/(r+drmin) * tol), drmin
        # minimum(maximum(abs(Q/dQ) * tol,dr0 * r), maximum(abs(Y/dY) * tol,dr0 * r))
        
        Q1 = Q + dQ*dr/2.
        Y1 = Y + dY*dr/2.
        r1 = r * (1. + dr/2.)
        
        dQ = RHSQQ(r1,k, omega, m)*Q1 + RHSQY(r1,k, omega, m) * Y1
        dY = RHSYQ(r1,k, omega, m)*Q1 + RHSYY(r1,k, omega, m) * Y1

        Qprev = Q ; Yprev = Y ; rprev = r

        Q+=dQ*dr
        Y+=dY*dr
        r+=r*dr
        # print("r = {:10.10f}".format(log10(r)), "; dr = ", dr)
        # ii = input("r")
        # maxQ = maximum(sqrt(Q.real**2+Q.imag**2), maxQ)

        if (r>=rstore): #  & ifplot:
            # print(r, Q, Y)
            if ifplot:
                print("r = ", r, "; dr = ", dr)
                fout.write(str(r)+" "+str(Q.real)+" "+str(Q.imag)+" "+str(Y.real)+" "+str(Y.imag)+"\n")
                fout.flush()
            rlist.append(r)
            qlist.append(Q)
            ylist.append(Y)
            rstore += drout
            
    Qlast = (Q - Qprev )/dr * (R0-rprev) + Qprev
    Ylast = (Y - Yprev )/dr * (R0-rprev) + Yprev

    rlist = asarray(rlist[1:])
    qlist = asarray(qlist[1:])
    ylist = asarray(ylist[1:])
    # Qlast /= sqrt(median(qlist.real**2+qlist.imag**2))
    
    # Ylast = (Y  - Yprev )/dr * rprev + Yprev

    if ifplot:

        fout.close()
        
        clf()
        fig=figure()
        plot(rlist, rlist*0., 'g--')
        plot(rlist, rlist*0.+1., 'g-')
        plot(rlist, rlist*0.+abs(m * (omega+m) / k / (sslope+1.)), 'g:')
        plot(rlist, rlist*0.-abs(m * (omega+m) / k / (sslope+1.)), 'g:')
        plot(rlist, qlist.real, 'k-', label=r'$\Re Q$')
        plot(rlist, qlist.imag, 'k:', label=r'$\Im Q$')
        plot(rlist, ylist.real, 'r-', label=r'$\Re Y$')
        plot(rlist, ylist.imag, 'r:', label=r'$\Im Y$')
        # ylim(-5.,5.)
        legend()
        xlabel(r'$\Omega r$')
        ylabel(r'$q(r)$, $y(r)$')
        fig.set_size_inches(4.,4.)
        savefig('QYplot.png')
        savefig('QYplot.pdf')
        
        rcr = sqrt(-m/k).real
        if (rcr > 0.) and (rcr < R0):
        
            yrat = -1j * (omega+m) * (m**2 - 2.*omega*k*rlist**2)/((2.-m)*k*rlist**2-m**2)/k
            clf()
            fig=figure()
            plot(rlist, rlist*0., 'g--')
            plot(rlist, (qlist/ylist).real, 'k-')
            plot(rlist, (qlist/ylist).imag, 'k:')
            plot(rlist, (yrat).real, 'r-')
            plot(rlist, (yrat).imag, 'r:')
            plot([rcr,rcr], [(qlist/ylist).imag.min(), (qlist/ylist).imag.max()], 'g:')
            xlabel(r'$\Omega r$')
            ylabel(r'$q(r)/y(r)$')
            fig.set_size_inches(4.,4.)
            savefig('QYrat.png')
            savefig('QYrat.pdf')

    print("k = ", k, ":  Q resid = ", (Qlast.real**2+Qlast.imag**2))
    # , ":  Y resid = ", log(Ylast.real**2+Ylast.imag**2))
    
    if isnan(Qlast):
        return [100.,100.]
        
    return [Qlast.real, Qlast.imag] # (Qlast.real**2+Qlast.imag**2) # +Ylast.real**2+Ylast.imag**2)

def onem(oar, m=1, k0 = [-1.737,-0.013], R0 = 1.):
    '''
    produces a series of solutions for a given array of omegas
    '''
    
    # k0 = [-1.5,2.0]
    
    rsol = None
    
    noar = size(oar)
    
    kre = zeros(noar)
    kim = zeros(noar)
    kre_plus = zeros(noar)
    kim_plus = zeros(noar)

    fout = open('ksoles_m'+str(m)+'.dat', 'w+')
    fout.write("omega  Re(k-)  Im(k-)  Re(k+)  Im(k+)\n")
    
    for i in arange(noar):
        res = root(onecurve, k0, args = (oar[i], m, R0))
        if res.success:
            res_plus = root(onecurve, [res.x[0], -res.x[1]], args = (oar[i], m, R0), tol=1e-8)
            kre[i] = res.x[0]
            kim[i] = res.x[1]
        # if res.success:
            if res_plus.success:
                print("omega = ", oar[i], ": k = ", res.x[0], "+",res.x[1], "i")
                fout.write(str(oar[i])+" "+str(res.x[0])+" "+str(res.x[1])+" "+str(res_plus.x[0])+" "+str(res_plus.x[1])+"\n")
                fout.flush()
                k0 = res.x
                k0[1] = res.x[1]
                kre_plus[i] = res_plus.x[0]
                kim_plus[i] = res_plus.x[1]
                onecurve(res.x, oar[i], m, R0, ifplot=True)
                filename = 'qysol_o'+str(oar[i])+'_m'+str(m)+'.dat'
                os.system('cp qysol.dat '+filename)
                if rsol is None:
                    lines = loadtxt(filename)
                    rsol = lines[1:,0].flatten()
                    nr = size(rsol)
                    qresol = zeros([nr, noar])
                    qimsol = zeros([nr, noar])
                    yresol = zeros([nr, noar])
                    yimsol = zeros([nr, noar])
                    qresol[:,0] = lines[1:,1] ; qimsol[:,0] = lines[1:,2]
                    yresol[:,0] = lines[1:,3] ; yimsol[:,0] = lines[1:,4]
                else:
                    lines = loadtxt(filename)
                    rsol1 = lines[1:,0].flatten()
                    qrefun = interp1d(rsol1, lines[1:,1], bounds_error = False, fill_value = "extrapolate")
                    qimfun = interp1d(rsol1, lines[1:,2], bounds_error = False, fill_value = "extrapolate")
                    yrefun = interp1d(rsol1, lines[1:,3], bounds_error = False, fill_value = "extrapolate")
                    yimfun = interp1d(rsol1, lines[1:,4], bounds_error = False, fill_value = "extrapolate")
                    qresol[:,i] = qrefun(rsol) ; qimsol[:,i] = qimfun(rsol)
                    yresol[:,i] = yrefun(rsol) ; yimsol[:,i] = yimfun(rsol)
            else:
                print("omega = ", oar[i], " diverged")
                kre_plus[i] = sqrt(-1)
                kim_plus[i] = sqrt(-1)
        else:
            kre[i] = sqrt(-1)
            kim[i] = sqrt(-1)

            # ii = input(res)
    fout.close()

    wneg = kim < 0.
    krenew = copy(kre)
    if wneg.sum() > 0:
        krenew[wneg] = kre_plus[wneg]
    kimnew = copy(kim)
    if wneg.sum() > 0:
        kimnew[wneg] = kim_plus[wneg]
        kre_plus[wneg] = kre[wneg]
        kim_plus[wneg] = kim[wneg]

    clf()
    fig = figure()
    plot(oar, krenew, 'k.')
    plot(oar, kimnew, 'rs')
    plot(oar, kre_plus, 'ok', mfc='none')
    plot(oar, kim_plus, 'rs', mfc='none')
    plot(oar, asarray(oar)*0., 'g:')
    xlabel(r'$\omega/\Omega$')
    ylabel(r'$k / \Omega$')
    fig.tight_layout()
    savefig('oreplot.png')
    
    clf()
    contourf(oar, rsol, qresol)
    plot(oar, asarray(oar)*0.+1., 'w:')
    xlabel(r'$\omega/\Omega$')
    ylabel(r'$\Omega r$')
    savefig('sols_Qre.png')

def Rvar(omega, m, norecalc = False):

    R0 = 2.0
    Rfac = 1.05
    Rmax = 20.
    
    k0 = [-0.3, -0.3]
    
    if norecalc:
    
        lines = loadtxt('kRsoles_m'+str(m)+'.dat')
        
        rlist = lines[:,0]
        klist_real = lines[:,1] ; klist_imag = lines[:,2]
    else:
    
        rlist = []
        klist_imag = []
        klist_real = []
    
        fout = open('kRsoles_m'+str(m)+'.dat', 'w+')
        fout.write("# Rout  Re(k)  Im(k)\n")
    
        while (R0 < Rmax):
            res = root(onecurve, k0, args = (omega, m, R0))
            if res.success:
                rlist.append(R0)
                klist_real.append(res.x[0])
                klist_imag.append(res.x[1])
                fout.write(str(R0)+" "+str(res.x[0])+" "+str(res.x[1])+"\n")
                fout.flush()
                k0 = res.x
            else:
                ii = input("lost")
                R0 = 1e6
            R0 *= Rfac
            print("R = ", R0)
        
        fout.close()
    
        rlist = asarray(rlist)
        klist_imag = asarray(klist_imag)
        klist_real = asarray(klist_real)

    clf()
    fig = figure()
    plot(rlist, klist_real, 'k.')
    plot(rlist, klist_imag, 'rx')
    plot(rlist, zeros(size(rlist)), 'g:')
    plot(rlist, .5/rlist**2, 'k:')
    xlabel(r'$\Omega R_{\rm out}$')
    ylabel(r'$k / \Omega$')
    xscale('log')
    fig.tight_layout()
    savefig('Rplot.png')
    
    sslope = sslopefun(omega, m)
    print("sigma = ", sslope)
    
    clf()
    fig = figure()
    plot(rlist, klist_real*rlist**2, 'k.')
    plot(rlist, klist_imag*rlist**2, 'rx')
    plot(rlist, zeros(size(rlist)), 'g:')
    xscale('log')
    xlabel(r'$\Omega R_{\rm out}$')
    ylabel(r'$R_{\rm out}^{2} k$')
    fig.tight_layout()
    savefig('kRplot.png')

    if not(norecalc):
        R0 = rlist[-1]
        onecurve([klist_real[-1], klist_imag[-1]], omega, m, R0, ifplot=True)
        os.system('cp qysol.dat qysol{:3.2f}.dat'.format(R0))
        onecurve([klist_real[0], klist_imag[0]], omega, m, rlist[0], ifplot=True)
        os.system('cp qysol.dat qysol{:3.2f}.dat'.format(rlist[0]))
        curvescompare('qysol{:3.2f}.dat'.format(rlist[0]), 'qysol{:3.2f}.dat'.format(R0))

def QYmin(omega, m, ifoptimize=True, imlog=False, R0=1., norecalc = False):

    if norecalc:
        lines = loadtxt('qymap.dat')
        k2real = lines[:,0]  ; k2imag = lines[:,1]
        reresid = lines[:,2]  ; imresid = lines[:,3]
        k2 = k2real + 1j * k2imag
        kreal = unique(k2real)
        kimag = unique(k2imag)
        
        nkreal = size(kreal) ; nkimag = size(kimag)
        reresid = reshape(reresid, (nkreal, nkimag))
        imresid = reshape(imresid, (nkreal, nkimag))
    else:
        nkreal = 50 ; nkimag = 21
        kreal_min = -5.0 ; kreal_max = 35.0
        kreal = (kreal_max-kreal_min) * arange(nkreal) / double(nkreal) + kreal_min
        if imlog:
            kimag_min = 0.001 ; kimag_max = 1.0
            kimag = (kimag_max/kimag_min) ** (arange(nkimag) / double(nkimag)) * kimag_min
        else:
            kimag_min = -2.001 ; kimag_max = 2.00
            kimag = (kimag_max-kimag_min) * arange(nkimag) / double(nkimag) + kimag_min
        
        k2 = zeros([nkreal, nkimag], dtype=complex)
    
        imresid = zeros([nkreal, nkimag])
        reresid = zeros([nkreal, nkimag])

        # ASCII output
        fout = open('qymap.dat', 'w+')
        fout.write('# Re(k)  Im(k)  Re(\Delta Q) Im(\Delta Q)\n')

        for kr in arange(nkreal):
            for ki in arange(nkimag):
                tmp = onecurve([kreal[kr], kimag[ki]], omega, m, R0=R0)
                imresid[kr, ki] = tmp[0]
                reresid[kr, ki] = tmp[1]
                k2[kr, ki] = kreal[kr] + 1j * kimag[ki]
                print("k = ", kreal[kr], "+", kimag[ki], "i:   ", reresid[kr, ki], " ", imresid[kr, ki])
                fout.write(str(kreal[kr])+" "+str(kimag[ki])+" "+str(reresid[kr, ki])+" "+str(imresid[kr, ki])+"\n")
                fout.flush()
        fout.close()
    wmin = (imresid**2+reresid**2).flatten().argmin()
    res0 = k2.flatten()[wmin]
    print("res0 = ", res0)

    if ifoptimize:
        res = root(onecurve, [res0.real, res0.imag], args = (omega, m, R0)) # , bounds=[(-2.,2.),(0.15,2.)], tol=1e-10)
        print(res)
        onecurve(res.x, omega, m, R0, ifplot=True)

    clf()
    fig =figure()
    # print("shape(imresid) = ", shape(imresid))
    pcolor(kreal, kimag, transpose(log10(imresid**2+reresid**2)))
    colorbar()
    contour(kreal, kimag, transpose(imresid), colors='w', levels=[0.], linestyles='-')
    contour(kreal, kimag, transpose(reresid), colors='w', levels=[0.], linestyles=':')
    if ifoptimize:
        if res.success:
            plot([res.x[0]], [res.x[1]], 'wx')
    xlabel(r'$\Re k$')
    ylabel(r'$\Im k$')
    if imlog:
        yscale('log')
    fig.tight_layout()
    fig.set_size_inches(kreal.max()-kreal.min(), kimag.max()-kimag.min())
    savefig('resmap.png')
    savefig('resmap.pdf')

#############################

def imtest():
    dx = 1e-2
    
    dxout = 0.1

    x = 0. ; q = 1.0 + 0.0j
    xstore = 0.0
    
    omega = 1.0 + 0.0j
    
    xmin = 5000. * pi
    xmax = 5010. * pi
    
    xlist = [] ; qlist = []
    
    while(x< xmax):
        dq = - 1j * omega * q
        
        q1 = q + dq * dx / 2.0
        
        dq = - 1j * omega * q1
        
        q += dq * dx
        x += dx
        
        if (x> xstore):
            xstore += dxout
            if (x > xmin):
                xlist.append(x)
                qlist.append(q)
            
    xlist = asarray(xlist)
    qlist = asarray(qlist)
            
    clf()
    fig = figure()
    subplot(311)
    plot(xlist, cos(xlist), 'r-')
    plot(xlist, qlist.real, 'k.')
    # xlabel(r'$x$')
    ylabel(r'$\Re q(x)$')
    subplot(312)
    plot(xlist, -sin(xlist), 'r-')
    plot(xlist, qlist.imag, 'k.')
    # xlabel(r'$x$')
    ylabel(r'$\Im q(x)$')
    subplot(313)
    plot(xlist, (qlist.real - cos(xlist)), 'k.')
    plot(xlist, (qlist.imag + sin(xlist)), 'r.')
    xlabel(r'$x$')
    ylabel(r'$\Delta q(x)$')
    fig.set_size_inches(10,6)
    savefig('imtest.png')

#  res = root(onecurve, [-2.614348900417678, 0.001], args = (0.4, 1, 1.), tol=1e-8)
# res = root(onecurve, [-1.7371693428009531, 0.], args = (0.4, 1, 1.), tol=1e-8)
# res = root(onecurve, [5.275542556065922, 0.], args = (0.4, 1, 1.), tol=1e-8)
# onecurve(res.x, 0.4,1,1., ifplot=True)
# QYmin(0.4,1)
# onecurve([1.25,0.], 0.4,1,1., ifplot=True) # k = m^2 / 2 omega Rout^2
# -2.61399535e+00,  1.31728768e-15
