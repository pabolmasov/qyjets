import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *

# visualization routines

formatsequence = ['k-', 'r:', 'g--', 'b-.', 'm--.']
nformats = size(formatsequence)

def rfun(z, psi, alpha, z0=10.):
    '''
    universal function to compute r(r, psi)
    '''
    chi = alpha * (1.+2.*alpha)/6.
    if alpha <= 0.:
        return sqrt(psi) * (z/z0)**alpha
    else:
        return z/z0 * sqrt((1.-sqrt(1.-4.*chi*psi*(z/z0)**(2.*(alpha-1.))))/2./chi)

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

def readnfiles(k, nblocks, ddir = 'paralpha0.0/', seq = False):
    '''
    reads a single snapshot stored as multiple dat files
    nblocks controls the number of mesh blocks we need to combine
    'seq' allows to read sequential data (no effect if nblocks >1)
    '''
    if nblocks <=1:
        if seq:
            return asciiread(ddir+'/pfiver{:05d}'.format(k)+'.dat')
        else:
            return asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat')
    else:
        z, x, Q, Er, Ez = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat')
        for j in arange(nblocks-1)+1:
            z1, x1, Q1, Er1, Ez1 = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(j)+'.dat')
            x = concatenate([x,x1])
            Q = concatenate([Q,Q1])
            Er = concatenate([Er,Er1])
            Ez = concatenate([Ez,Ez1])
        return z, x, Q, Er, Ez

def fiver_plotN(karray, nblocks=0, ddir = 'pfiver_alpha0.1/', p2d = False, alpha = 0.0, psislice = None):
    nk = size(karray)
    if nk <= 1:
        fiver_plot(karray[0], nblocks=nblocks)
    else:
        clf()
        fig, axs = plt.subplots(2)
        ctr  = 0
        for k in karray:
            z, x, Q, Er, Ez = readnfiles(k, nblocks, ddir=ddir, seq = (nblocks<=1))
            if p2d and (ctr == 0):
                zlist = []
                nz = size(karray)
                nx = size(x)
                q2 = zeros([nz, nx], dtype=complex)
                ez2 = zeros([nz, nx], dtype=complex)
                er2 = zeros([nz, nx], dtype=complex)
                qmax = zeros(nz, dtype = complex)
                ermax = zeros(nz, dtype = complex)
                ezmax = zeros(nz, dtype = complex)
            ztitle=r'$z = {:5.5f}$'.format(z)
            axs[0].plot(x, Q.real, formatsequence[ctr%nformats], label=ztitle)
            axs[1].plot(x, Q.imag, formatsequence[ctr%nformats], label=ztitle)
            if p2d:
                q2[ctr,:] = Q
                ez2[ctr,:] = Ez
                er2[ctr,:] = Er
                qmax[ctr] = abs(Q).max()
                ermax[ctr] = abs(Er).max()
                ezmax[ctr] = abs(Ez).max()
                zlist.append(z)
            ctr+=1
        axs[0].set_ylabel(r'$\Re Q$')
        axs[1].set_ylabel(r'$\Im Q$')
        axs[0].set_xlabel(r'$\psi$')
        axs[1].set_xlabel(r'$\psi$')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        if nk<10:
            axs[1].legend()
        fig.set_size_inches(5.,8.)
        fig.tight_layout()
        savefig(ddir+'/parfiverQs.png'.format(k))
        close()
        # Ez:
        clf()
        fig, axs = plt.subplots(2)
        ctr=0
        for k in karray:
            # ztitle=r'$z = {:5.5f}$'.format(z)
            axs[0].plot(x, ez2[ctr,:].real, formatsequence[ctr%nformats], label=zlist[ctr])
            axs[1].plot(x, ez2[ctr,:].imag, formatsequence[ctr%nformats], label=zlist[ctr])
            ctr+=1
        axs[0].set_ylabel(r'$\Re E_z$')
        axs[1].set_ylabel(r'$\Im E_z$')
        axs[0].set_xlabel(r'$\psi$')
        axs[1].set_xlabel(r'$\psi$')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        if nk < 10:
            axs[1].legend()
        fig.set_size_inches(5.,8.)
        fig.tight_layout()
        savefig(ddir+'/parfiverEzs.png'.format(k))
        # Er:
        clf()
        fig, axs = plt.subplots(2)
        ctr=0
        for k in karray:
            # ztitle=r'$z = {:5.5f}$'.format(z)
            axs[0].plot(x, er2[ctr,:].real, formatsequence[ctr%nformats], label=zlist[ctr])
            axs[1].plot(x, er2[ctr,:].imag, formatsequence[ctr%nformats], label=zlist[ctr])
            ctr+=1
        axs[0].set_ylabel(r'$\Re E_r$')
        axs[1].set_ylabel(r'$\Im E_r$')
        axs[0].set_xlabel(r'$\psi$')
        axs[1].set_xlabel(r'$\psi$')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        if nk < 10:
            axs[1].legend()
        fig.set_size_inches(5.,8.)
        fig.tight_layout()
        savefig(ddir+'/parfiverErs.png'.format(k))
        if p2d: # 2D plotting
            x2, z2 = meshgrid(x, zlist)
            clf()
            pcolor(x, zlist, log10(abs(q2))) # assuming x is the same
            cb = colorbar()
            cb.set_label(r'$\log_{10}|Q|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            savefig(ddir+'/Qabs.png')
            clf()
            pcolor(x, zlist, log10(abs(ez2))) # assuming x is the same
            cb = colorbar()
            cb.set_label(r'$\log_{10}|E_z|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            savefig(ddir+'/Ezabs.png')
            clf()
            pcolor(x, zlist, log10(abs(er2))) # assuming x is the same
            cb = colorbar()
            cb.set_label(r'$\log_{10}|E_r|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            savefig(ddir+'/Erabs.png')

    # growth curves:
    clf()
    plot(zlist, qmax, formatsequence[0], label=r'$\max |Q|$')
    plot(zlist, ezmax, formatsequence[1], label=r'$\max |E_z|$')
    plot(zlist, ermax, formatsequence[2], label=r'$\max |E_r|$')
    xlabel(r'$z$')
    yscale('log')
    xscale('log')
    legend()
    savefig(ddir+'/growthcurve.png')

    if psislice is not None:
        z = asarray(zlist)
        k = -2.6139953464141414 # kostylj
        wpsi = abs(x-psislice).argmin()
        clf()
        fig = figure()
        plot(zlist, (q2[0,wpsi] * exp(1.j * k * (z-z[0]))).real, 'r:')
        plot(zlist, (q2[0,wpsi] * exp(1.j * k * (z-z[0]))).imag, 'r--')
        plot(zlist, abs(q2)[:,wpsi], 'k-')
        plot(zlist, (q2.real)[:,wpsi], 'k:')
        plot(zlist, (q2.imag)[:,wpsi], 'k--')
        fig.set_size_inches(15.,5.)
        xlabel(r'$z$')
        savefig(ddir+'/zsliceQ.png')
        clf()
        fig = figure()
        plot(zlist, (er2[0,wpsi] * exp(1.j * k * (z-z[0]))).real, 'r:')
        plot(zlist, (er2[0,wpsi] * exp(1.j * k * (z-z[0]))).imag, 'r--')
        plot(zlist, abs(er2)[:,wpsi], 'k-')
        plot(zlist, (er2.real)[:,wpsi], 'k:')
        plot(zlist, (er2.imag)[:,wpsi], 'k--')
        fig.set_size_inches(15.,5.)
        xlabel(r'$z$')
        savefig(ddir+'/zsliceEr.png')


def fiver_plot(k, nblocks=0, ddir = 'paralpha0.0'):

    z, x, Q, Er, Ez = readnfiles(k, nblocks, ddir = ddir, seq = (nblocks<=1))

    ztitle=r'$z = {:5.5f}$'.format(z)

    clf()
    fig =figure()
    plot(x, Q.real, 'k-')
    plot(x, Q.imag, 'k:')
    plot(x, x*0., 'g--')
    xlabel(r'$\psi$')
    ylabel(r'$Q(\psi)$')
    title(ztitle)
    xscale('log')
    fig.tight_layout()
    savefig(ddir+'/parfiverQ{:05d}.png'.format(k))
    clf()
    fig =figure()
    plot(x, Ez.real, 'k-')
    plot(x, Ez.imag, 'k:')
    plot(x, x*0., 'g--')
    xlabel(r'$\psi$')
    ylabel(r'$E_z(\psi)$')
    title(ztitle)
    xscale('log')
    fig.tight_layout()
    savefig(ddir+'/parfiverEz{:05d}.png'.format(k))
    clf()
    fig =figure()
    plot(x, Er.real, 'k-')
    plot(x, Er.imag, 'k:')
    plot(x, x*0., 'g--')
    xlabel(r'$\psi$')
    ylabel(r'$E_r(\psi)$')
    title(ztitle)
    xscale('log')
    fig.tight_layout()
    savefig(ddir+'/parfiverEr{:05d}.png'.format(k))
    close()

def comparer(npar, npsi, nblocks=2):
    '''
    compares parallel and sequential data
    inputs:
    npar -- No of the parallel snapshot
    npsi -- No of the sequential snapshot
    '''
    zpar, xpar, Qpar, Erpar, Ezpar = readnfiles(npar, nblocks, ddir = 'paralpha0.0')
    zpar, xpar, Qpar, Erpar, Ezpar = readnfiles(npar, 0, ddir = 'pfiver_alpha0.0', seq=True)
    
    print(zpar, ' = ', zpsi)
    
    ztitle_par=r'parallel $z = {:5.5f}$'.format(zpar)
    ztitle_psi=r'sequential $z = {:5.5f}$'.format(zpar)

    clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(xpar, Qpar.real, formatsequence[0], label=ztitle_par)
    axs[1].plot(xpar, Qpar.imag, formatsequence[0], label=ztitle_par)
    axs[0].plot(xpsi, Qpsi.real, formatsequence[1], label=ztitle_psi)
    axs[1].plot(xpsi, Qpsi.imag, formatsequence[1], label=ztitle_psi)
    axs[0].set_ylabel(r'$\Re Q$')
    axs[1].set_ylabel(r'$\Im Q$')
    axs[0].set_xlabel(r'$\psi$')
    axs[1].set_xlabel(r'$\psi$')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[1].legend()
    fig.set_size_inches(5.,8.)
    fig.tight_layout()
    savefig('comQs.png')

    clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(xpar, Ezpar.real, formatsequence[0], label=ztitle_par)
    axs[1].plot(xpar, Ezpar.imag, formatsequence[0], label=ztitle_par)
    axs[0].plot(xpsi, Ezpsi.real, formatsequence[1], label=ztitle_psi)
    axs[1].plot(xpsi, Ezpsi.imag, formatsequence[1], label=ztitle_psi)
    axs[0].set_ylabel(r'$\Re E_z$')
    axs[1].set_ylabel(r'$\Im E_z$')
    axs[0].set_xlabel(r'$\psi$')
    axs[1].set_xlabel(r'$\psi$')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[1].legend()
    fig.set_size_inches(5.,8.)
    fig.tight_layout()
    savefig('comEzs.png')

def qysol_plot(infile):
    
    lines = loadtxt(infile)
    omega, m, rout, kre, kim = lines[0,:]
    
    # print(lines[0,:])
    
    r = lines[1:, 0]
    Q = lines[1:, 1] + lines[1:, 2] * 1.j
    Y = lines[1:, 3] + lines[1:, 4] * 1.j
    
    clf()
    # plot(r, abs(Q), 'g--', label=r'$|Q|$')
    # plot(r, -abs(Q), 'g--', label=r'$-|Q|$')
    plot(r, Q.real, 'k-', label=r'$\Re Q$')
    plot(r, Q.imag, 'k:', label=r'$\Im Q$')
    xscale('log')
    legend()
    ylabel(r'$Q$')
    xlabel(r'$r$')
    savefig(infile+'_Q.png')

    clf()
    #plot(r, abs(Y), 'g--', label=r'$|Y|$')
    # plot(r, -abs(Y), 'g--', label=r'$-|Y|$')
    plot(r, Y.real, 'k-', label=r'$\Re Y$')
    plot(r, Y.imag, 'k:', label=r'$\Im Y$')
    xscale('log')
    legend()
    ylabel(r'$Y$')
    xlabel(r'$r$')
    savefig(infile+'_Y.png')
    close()
