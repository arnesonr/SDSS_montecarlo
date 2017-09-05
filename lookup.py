import numpy as n
import matplotlib as m
# The following 2 lines are necessary to make the
# GUI work right, at least for me. YMMV!
m.use('TkAgg')
m.interactive(True)
from matplotlib import pyplot as p
from matplotlib import cm
import scipy as s
from scipy import optimize as opt
import pyfits as pf
import lensdemo_funcs as ldf
import psf_funcs as g
from scipy import signal

def lookup(par):
    """
    PURPOSE: Calculate the flux received in a SDSSIII fiber over the intrinsic flux as a
    function of impact parameter, Einstein radius, axis ratio, size, etc...

    USAGE: f_f/f_i = lookup(par[b, q_l, P.A_l, amp, sigma_s, xcen_s, ycen_s,q_s, P.A_s])

    ARGUMENTS:
    pars:
       par[0]: Einstein Radius
       par[1]: Axis ratio of lens
       par[2]: Position angle of lens (c.c.w. major-axis rotation w.r.t. x-axis)
       par[3]: Amplitude of source galaxy
       par[4]: Intermediate-axis sigma of source galaxy
       par[5]: x coordinate of source galaxy
       par[6]: y coordinate of source galaxy
       par[7]: Axis ratio of source
       par[8]: Position angle of source (c.c.w. major-axis rotation w.r.t. x-axis)

    RETURNS:  Ratio of flux in fiber w.r.t intrinsic flux

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """
    #Define the pixel scale to be 0.01"/pixel
    
    myargs = {'interpolation': 'nearest', 'origin': 'lower','cmap': cm.gray, 'hold': False}

    psf = g.gauss2d(800,150)

    nx = 1600.
    ny = 1600.
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.

    #lpar_guess = n.asarray([b, xcen, ycen, q, P.A.])
    lpar_guess = n.asarray([par[0], 0.0, 0.0, par[1], par[2]])
    #gpar_guess = n.asarray([amp., sigma, xcen, ycen, q, P.A.])
    gpar_guess = n.asarray([par[3], par[4], par[5], par[6], par[7], par[8]])
    xg, yg = ldf.sie_grad(ximg,yimg, lpar_guess)
    lmodel = ldf.gauss_2d(ximg-xg, yimg-yg, gpar_guess)
    #lmodel = signal.fftconvolve(lmodel, psf, mode='same')
    #lmodel = lmodel[1:1601,1:1601]

    fiber = yimg
    for i in range (0,1600):
        for j in range (0,1600):
            fiber[i,j]=n.sqrt(fiber[i,j]**2+(j-800.)**2)

    for i in range (0,1600):
        for j in range (0,1600):
            if fiber[i,j] <= 100:
                fiber[i,j]=1.0
            else: fiber[i,j]=0.0

    fiber = signal.fftconvolve(fiber,psf,mode='same')
    fiber = fiber[1:1601,1:1601]

    f_f = n.sum(lmodel*fiber)
    #calculate the intrinisic flux (i.e. b=0)
    lpar_guess[0]=0.0
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    xg, yg = ldf.sie_grad(ximg,yimg, lpar_guess)
    lmodel2 = ldf.gauss_2d(ximg-xg, yimg-yg, gpar_guess)
    #lmodel2 = signal.fftconvolve(lmodel2, psf, mode='same')
    #lmodel2 = lmodel2[1:1601,1:1601]
    f_i = n.sum(lmodel2*fiber)

    mag = f_f/f_i
    #p.imshow(n.hstack((lmodel*fiber,lmodel2*fiber)),**myargs)
    return mag
