import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *

formatsequence = ['k-', 'r:', 'g--', 'b-.', 'm--.']
nformats = size(formatsequence)

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

def readnfiles(k, nblocks, ddir = 'paralpha0.0/'):

    if nblocks <=1:
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

def fiver_plotN(karray, nblocks=0, ddir = 'paralpha0.0/', p2d = False):
    nk = size(karray)
    if nk <= 1:
        fiver_plot(karray[0], nblocks=nblocks)
    else:
        clf()
        fig, axs = plt.subplots(2)
        ctr  = 0
        for k in karray:
            z, x, Q, Er, Ez = readnfiles(k, nblocks, ddir=ddir)
            if p2d and (ctr == 0):
                zlist = []
                nz = size(karray)
                nx = size(x)
                q2 = zeros([nz, nx])
            ztitle=r'$z = {:5.5f}$'.format(z)
            axs[0].plot(x, Q.real, formatsequence[ctr%nformats], label=ztitle)
            axs[1].plot(x, Q.imag, formatsequence[ctr%nformats], label=ztitle)
            if p2d:
                q2[ctr,:] = abs(Q)
                zlist.append(z)
            ctr+=1

        axs[0].set_ylabel(r'$\Re Q$')
        axs[1].set_ylabel(r'$\Im Q$')
        axs[0].set_xlabel(r'$\psi$')
        axs[1].set_xlabel(r'$\psi$')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        if size(karray)<10:
            axs[1].legend()
        fig.set_size_inches(5.,8.)
        fig.tight_layout()
        savefig(ddir+'/parfiverQs.png'.format(k))
        close()
        if p2d:
            clf()
            pcolor(x, zlist, log10(q2)) # assuming x is the same
            cb = colorbar()
            cb.set_label(r'$\log_{10}|Q|$')
            # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            savefig(ddir+'/Qabs.png')
        # Ez:
        clf()
        fig, axs = plt.subplots(2)
        ctr=0
        for k in karray:
            z, x, Q, Er, Ez = readnfiles(k, nblocks)
            ztitle=r'$z = {:5.5f}$'.format(z)
            axs[0].plot(x, Ez.real, formatsequence[ctr%nformats], label=ztitle)
            axs[1].plot(x, Ez.imag, formatsequence[ctr%nformats], label=ztitle)
            ctr+=1
        axs[0].set_ylabel(r'$\Re Ez$')
        axs[1].set_ylabel(r'$\Im Ez$')
        axs[0].set_xlabel(r'$\psi$')
        axs[1].set_xlabel(r'$\psi$')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[1].legend()
        fig.set_size_inches(5.,8.)
        fig.tight_layout()
        savefig(ddir+'/parfiverEzs.png'.format(k))


def fiver_plot(k, nblocks=0, ddir = 'paralpha0.0'):

    z, x, Q, Er, Ez = readnfiles(k, nblocks, ddir = ddir)

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

    zpar, xpar, Qpar, Erpar, Ezpar = readnfiles(npar, nblocks, ddir = 'paralpha0.0')

    zpsi, xpsi, Qpsi, Erpsi, Ezpar =asciiread('pfiver_alpha0.0/pfiver{:05d}'.format(npsi)+'.dat')
    
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

