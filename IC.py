from numpy import *


def splinefun(x, x1, x2, x3, a):
    '''
    piecewise-polynomial IC exactly 0 at the boundaries
    '''
    nx = size(x)
    y = zeros(nx, dtype=complex128)
    ydiv = zeros(nx, dtype=complex128)
    
    w1 = (x<x2) * (x>x1)
    w2 = (x<x3) * (x>x2)
    
    xnorm1 = (x-x1)/(x2-x1)
    xnorm2 = (x-x3)/(x2-x3)

    y[w1] = (3.-2.*xnorm1[w1])*xnorm1[w1]**2
    y[w2] = (3.-2.*xnorm2[w2])*xnorm2[w2]**2

    ydiv[w1] = 6. * (1.-xnorm1[w1]) * xnorm1[w1] / (x2 - x1)
    ydiv[w2] = 6. * (1.-xnorm2[w2]) * xnorm2[w2] / (x2 - x3)

    return y * a, ydiv * a

