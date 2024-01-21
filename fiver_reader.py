import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *

# visualization routines

# TODO: make them readable!
omega = 0.4
m = 1
R0 = 1.0
z0 = 10.

#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]


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

def asciiread(fname, ifBY = False):

    f = open(fname, "r")
    s = f.readline() # header string
    z = double(s[s.find('=')+2:len(s)-1])
    f.close()
    
    # bulk of the data
    lines = loadtxt(fname)
    x = lines[:,0]
    if ifBY:
        bre = lines[:,1] ; bim = lines[:,2]
        yre = lines[:,3] ; yim = lines[:,4]
        return z, x, bre+1.j*bim, yre+1.j*yim
    else:
        qre = lines[:,1] ; qim = lines[:,2]
        erre = lines[:,3] ; erim = lines[:,4]
        ezre = lines[:,5] ; ezim = lines[:,6]
        return z, x, qre+1.j*qim, erre+1.j*erim, ezre+1.j*ezim

def readnfiles(k, nblocks, ddir = 'paralpha0.0/', seq = False, ifBY = False):
    '''
    reads a single snapshot stored as multiple dat files
    nblocks controls the number of mesh blocks we need to combine
    'seq' allows to read sequential data (no effect if nblocks >1)
    '''
    # TODO: par support for BY
    if nblocks <=1:
        if seq:
            if ifBY:
                return asciiread(ddir+'/pfiverBY{:05d}'.format(k)+'.dat', ifBY=ifBY)
            else:
                return asciiread(ddir+'/pfiver{:05d}'.format(k)+'.dat', ifBY=ifBY)
        else:
            return asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat', ifBY=ifBY)
    else:
        if ifBY:
            z, x, B, Y = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat', ifBY = True)
        else:
            z, x, Q, Er, Ez = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat', ifBY = False)
        for j in arange(nblocks-1)+1:
            if ifBY:
                z1, x1, B1, Y1 = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat', ifBY = True)
            else:
                z1, x1, Q1, Er1, Ez1 = asciiread(ddir+'/par{:05d}'.format(k)+'.{:03d}'.format(0)+'.dat', ifBY = False)
            x = concatenate([x,x1])
            if ifBY:
                B = concatenate([B, B1])
                Y = concatenate([Y, Y1])
            else:
                Q = concatenate([Q,Q1])
                Er = concatenate([Er,Er1])
                Ez = concatenate([Ez,Ez1])
        if BY:
            return z, x, B, Y
        else:
            return z, x, Q, Er, Ez
        
def fiver_plotN(karray, nblocks=0, ddir = 'pfiver_alpha0.1/', p2d = False, alpha = 0.0, psislice = None, ifBY = False):
    nk = size(karray)
    if nk <= 1:
        fiver_plot(karray[0], nblocks=nblocks)
    else:
        clf()
        fig, axs = plt.subplots(2)
        ctr  = 0
        for k in karray:
            z, x, Q, Er, Ez = readnfiles(k, nblocks, ddir=ddir, seq = (nblocks<=1))
            if ifBY:
                z, xf, B, Y = readnfiles(k, nblocks, ddir=ddir, seq = (nblocks<=1), ifBY = True)

            if p2d and (ctr == 0):
                zlist = []
                nz = size(karray)
                nx = size(x)
                q2 = zeros([nz, nx], dtype=complex)
                ez2 = zeros([nz, nx], dtype=complex)
                er2 = zeros([nz, nx], dtype=complex)
                if ifBY:
                    b2 = zeros([nz, nx+1], dtype=complex)
                    y2 = zeros([nz, nx+1], dtype=complex)
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
                if ifBY:
                    b2[ctr,:] = B
                    y2[ctr,:] = Y
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
        if nk < 100:
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
            fig = figure()
            pcolor(x, zlist, (abs(q2))) # assuming x is the same
            cb = colorbar()
            # cb.set_label(r'$|Q|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            title(r'$|Q|$')
            fig.set_size_inches(5.,8.)
            fig.tight_layout()
            savefig(ddir+'/Qabs.png')
            clf()
            fig = figure()
            pcolor(x, zlist, (abs(ez2))) # assuming x is the same
            title(r'$|E_z|$')
            cb = colorbar()
            # cb.set_label(r'$|E_z|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            fig.set_size_inches(5.,8.)
            fig.tight_layout()
            savefig(ddir+'/Ezabs.png')
            clf()
            pcolor(x, zlist, (abs(er2))) # assuming x is the same
            cb = colorbar()
            # cb.set_label(r'$|E_r|$')
            contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
            title(r'$|E_r|$')
            # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
            xlabel(r'$\psi$')
            ylabel(r'$z$')
            
            fig.set_size_inches(5.,8.)
            fig.tight_layout()
            savefig(ddir+'/Erabs.png')
            if ifBY:
                clf()
                fig = figure()
                pcolor(xf, zlist, (abs(b2))) # assuming x is the same
                cb = colorbar()
                cb.set_label(r'$|B|$')
                contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
                # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
                xlabel(r'$\psi$')
                ylabel(r'$z$')
                fig.set_size_inches(5.,8.)
                fig.tight_layout()
                savefig(ddir+'/Babs.png')
                clf()
                fig=figure()
                pcolor(xf, zlist, (abs(y2))) # , vmin = -2.0, vmax = 0.5) # assuming x is the same
                cb = colorbar()
                cb.set_label(r'$|Y|$')
                contour(x2, z2, rfun(z2, x2, alpha, z0 = zlist[0]), colors='w') #
                # contour(exp(psi2), z2, rfun(z2, psi2), colors='w')
                #plot(x, z0 + x / (2.*sqrt((omega+m)/omega)) , 'k:')
                #plot(x, z0 + (x[-1]-x) / (2.*sqrt((omega+m)/omega)) , 'k:')
                # plot(x, z0 + (x[-1]-x) / (200.) , 'r:')
                # plot(x, z0 + (x[0]**1.5-x**1.5)*3./sqrt(2.*omega), 'k:')
                # plot(x, z0 - (log(x)+log(x.min()))/double(1e3), 'k:')
                # plot(x, z0 + (log(x)-2.*log(x.min()))/double(1e3), 'k:')
                # plot(x, z0 + 1e-4 * arange(size(x)), 'k:')
                xlabel(r'$\psi$')
                ylabel(r'$z$')
                ylim(z2.min(), z2.max())
                fig.set_size_inches(5.,8.)
                fig.tight_layout()
                savefig(ddir+'/Yabs.png')

    # growth curves:
    clf()
    fig = figure()
    plot(zlist, qmax, formatsequence[0], label=r'$\max |Q|$')
    plot(zlist, ezmax, formatsequence[1], label=r'$\max |E_z|$')
    plot(zlist, ermax, formatsequence[2], label=r'$\max |E_r|$')
    xlabel(r'$z$')
    yscale('log')
    # xscale('log')
    legend()
    fig.set_size_inches(8.,4.)
    fig.tight_layout()
    savefig(ddir+'/growthcurve.png')
    close('all')
    
    if ifBY:
        # ezhalf = (ez2[:,0]+ez2[:,1])/2.
        # erhalf = (er2[:,0]+er2[:,1])/2.
        ezhalf = ez2[:,0] # -(ez2[:,1]-ez2[:,0])/2. ;
        erhalf = er2[:,0] # -(er2[:,1]-er2[:,0])/2.

        # yhalf = (y2[:,0]+y2[:,1])/2.
        # bhalf = (b2[:,0]+b2[:,1])/2.

        bhalf = b2[:,0] ; yhalf = y2[:,0]

        ycheck = - 1.j * (z/z0)**(-alpha) * (ezhalf + alpha/(z/z0)**(1.-alpha)*erhalf)
        bcheck = 2.j*(1.-omega)/m * (ezhalf + alpha/z**(1.-alpha)*erhalf) - 1.j*ezhalf

        print("Y0 = ", ycheck[0])
        print("Y0half = ", yhalf[0])

        absnormY = maximum(abs(ycheck), abs(y2[:,0]))
        absnormB = maximum(abs(bcheck), abs(b2[:,0]))

        # absnormB = 1. ; absnormY = 1.

        clf()
        fig = figure()
        plot(zlist, (yhalf-ycheck).real/absnormY, 'k:', label=r'$\Re (Y-Y_0)/\max(|Y|, |Y_0|)$')
        plot(zlist, (yhalf-ycheck).imag/absnormY, 'k--', label=r'$\Im (Y-Y_0)/\max(|Y|, |Y_0|)$')
        plot(zlist, (bhalf-bcheck).real/absnormB, 'r:', label=r'$\Re (B-B_0)/\max(|B|, |B_0|)$')
        plot(zlist, (bhalf-bcheck).imag/absnormB, 'r--', label=r'$\Im (B-B_0)/\max(|B|, |B_0|)$')
        legend()
        fig.set_size_inches(15.,5.)
        title(r'$\alpha = {:1.3f}$'.format(alpha))
        xlabel(r'$z$')
        savefig(ddir+'/BCcheck.png')

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
        title(r'$\psi = {:1.3f}$'.format(psislice))
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
        title(r'$\psi = {:1.3f}$'.format(psislice))
        xlabel(r'$z$')
        savefig(ddir+'/zsliceEr.png')
        clf()
        fig = figure()
        plot(zlist, (ez2[0,wpsi] * exp(1.j * k * (z-z[0]))).real, 'r:')
        plot(zlist, (ez2[0,wpsi] * exp(1.j * k * (z-z[0]))).imag, 'r--')
        plot(zlist, abs(ez2)[:,wpsi], 'k-')
        plot(zlist, (ez2.real)[:,wpsi], 'k:')
        plot(zlist, (ez2.imag)[:,wpsi], 'k--')
        fig.set_size_inches(15.,5.)
        title(r'$\psi = {:1.3f}$'.format(psislice))
        xlabel(r'$z$')
        savefig(ddir+'/zsliceEz.png')


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

def comparseq(n1, n2, dir1, dir2):
    '''
    compares two sequential files
    '''
    z1, x1, Q1, Er1, Ez1 = readnfiles(n1, 0, ddir = dir1, seq=True)
    z2, x2, Q2, Er2, Ez2 = readnfiles(n2, 0, ddir = dir2, seq=True)

    ztitle1=r': $z1 = {:5.5f}$'.format(z1)
    ztitle2=r': $z2 = {:5.5f}$'.format(z2)

    clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(x1, Q1.real, formatsequence[0], label=ztitle1)
    axs[1].plot(x1, Q1.imag, formatsequence[0], label=ztitle1)
    axs[0].plot(x2, Q2.real, formatsequence[1], label=ztitle2)
    axs[1].plot(x2, Q2.imag, formatsequence[1], label=ztitle2)
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
    axs[0].plot(x1, Er1.real, formatsequence[0], label=ztitle1)
    axs[1].plot(x1, Er1.imag, formatsequence[0], label=ztitle1)
    axs[0].plot(x2, Er2.real, formatsequence[1], label=ztitle2)
    axs[1].plot(x2, Er2.imag, formatsequence[1], label=ztitle2)
    axs[0].set_ylabel(r'$\Re E_r$')
    axs[1].set_ylabel(r'$\Im E_r$')
    axs[0].set_xlabel(r'$\psi$')
    axs[1].set_xlabel(r'$\psi$')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[1].legend()
    fig.set_size_inches(5.,8.)
    fig.tight_layout()
    savefig('comErs.png')
    clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(x1, Ez1.real, formatsequence[0], label=ztitle1)
    axs[1].plot(x1, Ez1.imag, formatsequence[0], label=ztitle1)
    axs[0].plot(x2, Ez2.real, formatsequence[1], label=ztitle2)
    axs[1].plot(x2, Ez2.imag, formatsequence[1], label=ztitle2)
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

# for the harmonic solution:

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

def onem_plot(m=1):
    infile = 'ksoles_m'+str(m)+'.dat'
    lines = loadtxt(infile)
    omega = lines[:,0]
    kre = lines[:,1]
    kim = lines[:,2]
    
    clf()
    fig = figure()
    plot(omega, kre, 'k.')
    plot(omega, kim, 'rs')
    # plot(oar, kre_plus, 'ok', mfc='none')
    # plot(oar, kim_plus, 'rs', mfc='none')
    plot(omega, omega*0., 'g:')
    plot(omega, m-kre, 'b:')
    xlabel(r'$\omega/\Omega$')
    ylabel(r'$k / \Omega$')
    fig.tight_layout()
    savefig('oreplot.png')

