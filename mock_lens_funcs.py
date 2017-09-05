import numpy as n
import distance_funcs as df
import numpy.random as r
from matplotlib import pyplot as p
import matplotlib as m
import gblob as gb
from scipy import signal
import psf_funcs as blur
from matplotlib import cm

def ellip(N):
    """
    PURPOSE: Randomly draws ellipticities (between 0.0 and 0.95) that follows
             f(E) = -(E^.4-.95^.4)*E^.4 [~Holden et al. 2009]

    USAGE: array = ellip(n)
    
    ARGUMENTS:
              n: number of objects to select from the probability function

    RETURNS:  An array of n objects with values of ellipticity

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    E = list()

    while n.size(E) < N:
        x=r.uniform(0.,.95)
        y=r.uniform(0.,.24)
        
        if y < x**.4*(-x**.4+.95**.4):
            E.append(x)

    return E

def einstein_r(N, b, sig):
    """
    PURPOSE: Randomly draws an einstein radius that follows p(log b) ~ exp[-(log b - m)^2/ 2*s^2)]
             where m = log(b), s = sig/(b*ln(10))

    USAGE: array = einstein_r(n)
    
    ARGUMENTS:
              N: number of objects to select from the probability function
              b: average Einstein radius in arcsec
              sig: sigma of the Einstein radii distribution in arcsec

    RETURNS:  An array of n objects with values of Einstein radius in pixels (0.01 arcsec/pixel)

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    b = b/0.01
    m = n.log10(b)
    sig = sig/0.01
    s = sig/(b*n.log(10))
    er = r.normal(m, s, N)
    er = 10**er
    return er

def gamma(N, g, sig):
    """
    PURPOSE: Randomly draws a mass profile from p(g) ~ exp[-(g - m)^2/ 2*s^2)]
             where m = ave. g, s = sig g

    USAGE: array = gamma(N,g,sig)
    
    ARGUMENTS:
              N: number of objects to select from the probability function
              g: average mass profile parameter
              sig: sigma of the mass profile distribution

    RETURNS:  An array of n objects with values of gamma

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    x=list()
    while n.size(x) < N:
        a = r.normal(g, sig, 1)
        if 0.0 < a < 2.0:
            x.append(a[0])
    return x

def s_size(N):
    """
    PURPOSE: Randomly selects a source galaxy radius from the SLACS XI paper (Newton et al.)
             Assuming a uniform distribution between minimum and maximum source radius

    USAGE: radius = s_size(n)

    ARGUMENTS:  n: number of galaxy sizes to return

    RETURNS:  A source galaxy radius randomly selected from the SLACS XI paper. in pixels
              (0.01 arcsec/pixel)

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    Rs = r.uniform(3.0,100.,N)
    return Rs

def exp_disk(x, y, par):
    """
    NAME: exp_disk

    PURPOSE: Implement an exponential disk function

    USAGE: z = exp_disk(x, y, fullpar)

    ARGUMENTS:
      x, y: vecors or images of coordinates;
            should be matching numpy ndarrays
      fullpar: vector of parameters, defined as follows:
        fullpar[0]: amplitude
        fullpar[1]: intermediate-axis sigma
        fullpar[2]: x-center
        fullpar[3]: y-center
        fullpar[4]: axis ratio
        fullpar[5]: c.c.w. major-axis rotation w.r.t. x-axis
      Additional exponential disk components can be added via
      fullpar[6:12], fullpar[12:18], etc.

    RETURNS: Exponential disk evaluated at x-y coords

    NOTE: amplitude = 1 is not normalized, but rather has max = 1

    WRITTEN: Ryan A. Arneson, U. of Utah, 2011
    """
    
    (xnew,ynew) = gb.xy_rotate(x, y, par[2], par[3], par[5])
    r_ell = n.sqrt(((xnew**2)*par[4] + (ynew**2)/par[4])) / n.abs(par[1])
    z = par[0] * n.exp(-r_ell)
    return z
    
def exp_disk_multi(x, y, fullpar):
    """
    NAME: exp_disk

    PURPOSE: Implement an exponential disk function

    USAGE: z = exp_disk(x, y, fullpar)

    ARGUMENTS:
      x, y: vecors or images of coordinates;
            should be matching numpy ndarrays
      fullpar: vector of parameters, defined as follows:
        fullpar[0]: amplitude
        fullpar[1]: intermediate-axis sigma
        fullpar[2]: x-center
        fullpar[3]: y-center
        fullpar[4]: axis ratio
        fullpar[5]: c.c.w. major-axis rotation w.r.t. x-axis
      Additional exponential disk components can be added via
      fullpar[6:12], fullpar[12:18], etc.

    RETURNS: Exponential disk evaluated at x-y coords

    NOTE: amplitude = 1 is not normalized, but rather has max = 1

    WRITTEN: Ryan A. Arneson, U. of Utah, 2011
    """
    n_each = 6
    z = 0. * x
    nexp = len(fullpar) / n_each
    for i in range(nexp):
        par = fullpar[i*n_each:(i+1)*n_each]
        (xnew,ynew) = gb.xy_rotate(x, y, par[2], par[3], par[5])
        r_ell = n.sqrt(((xnew**2)*par[4] + (ynew**2)/par[4])) / n.abs(par[1])
        z += par[0] * n.exp(-r_ell)
    return z

def lenses(par): #Num, bave, sig_b, gave, sig_g, rfib):
    """
    PURPOSE: Generates mock lensing galaxies with randomly selected Einstein
             radii, axis ratios, and mass profiles (see mock_lens_funcs.py) at a
             fixed redshift (Zl = 0.2) And source galaxies at a fixed redshift
             (zs = 0.8) whose position is uniformly distributed in the fiber and size
             is randomly selected

    USAGE: list = lenses([Num, bave, sig_b, gave, sig_g, rfib])

    ARGUMENTS:
              par[0] Num: integer number of lens galaxies to randomly generate
              par[1] bave: average Einstein radius in arcsec
              par[2] sig_b: sigma of the Einstein radii distribution in arcsec
              par[3] gave: average mass profile parameter
              par[4] sig_g: sigma of the mass profile distribution
              par[5] rfib: radius of the fiber in arcsec

    RETURNS:  A list of the Einstein radii, axis ratios, mass profiles, source
              position in the fiber (r,theta), and source size
              i.e. [b, e, g, theta, rho, rs]

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    
    #initialize redshift and list of objects
    a = list()
    #randomly select Einstein radii, axis ratios, and mass profiles
    e = ellip(par[0])
    b = einstein_r(par[0], par[1], par[2])
    g = gamma(par[0], par[3], par[4])
    #generate rho and theta array for source galaxy location
    rho = n.zeros(par[0])
    theta = n.zeros(par[0])
    #randomly pick rho and theta in the fiber for the source
    for i in range (0,par[0]):
        #radial position (rho) of galaxy is in units of pixels from fiber center
        #0.01 arcsec/pixel
        rho[i] = (par[5]/0.01) * n.sqrt(r.uniform(0,1))
        #theta position is in degrees c.c.w. from x-axis (East)
        theta[i] = r.uniform(0.0,2*n.pi)
    #randomly select the size of the source
    rs = s_size(par[0])
    #write the parameter to the list
    for i in range (0,par[0]):
        a.append([b[i],e[i],g[i],theta[i],rho[i],rs[i]])
    return a

def magnification(lenslist, rfib, seeing):
    """
    PURPOSE: Calculate the flux received in a SDSSIII fiber over the intrinsic flux as a
    function of impact parameter, Einstein radius, axis ratio, sourc size, and lens galaxy
    mass profile, and seeing effects

    USAGE: array = magnification(lenslist, rfib, seeing)

    ARGUMENTS:
    lenslist:
       lenslist [i][0]: Einstein Radius
       lenslist [i][1]: ellipticity of lens
       lenslist [i][2]: power-law index 'gamma': surface density propto R^-gamma
                gamma = 1 is isothermal (r^-2 in 3D);
                gamma > 1 is steeper than isothermal
                gamma < i is shallower than isothermal.
       lenslist [i][3]: Position angle of source (c.c.w. major-axis rotation w.r.t. x-axis)
       lenslist [i][4]: radial position of source galaxy in pixels 
       lenslist [i][5]: Effective radius of source galaxy (major axis)
    rfib:  Fiber radius in arcsec
    seeing: FWHM of seeing PSF in arcsec

    RETURNS:  array of ratio of flux in fiber w.r.t intrinsic flux

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    #
    #Define the pixel scale to be 0.01"/pixel
    #
    #Define lists of lens variables
    how_many = n.size(lenslist)/6
    b=list()
    e=list()
    g=list()
    theta=list()
    rho=list()
    rs=list()
    
    #generate the psf (must be a matching 2d array with lmodel and smodel)
    psf = blur.gauss2d(800,(seeing/0.01))

    nx = 1600.
    ny = 1600.
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    #generate the fiber mask
    fiber = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    for i in range (0,1600):
        for j in range (0,1600):
            fiber[i,j]=n.sqrt(fiber[i,j]**2+(j-800.)**2)

    for i in range (0,1600):
        for j in range (0,1600):
            if fiber[i,j] <= (rfib/0.01):
                fiber[i,j]=1.0
            else: fiber[i,j]=0.0
    #convolve the fiber with the psf
    fiber = signal.fftconvolve(fiber,psf,mode='same')
    fiber = fiber[1:1601,1:1601]
    
    #Find all the parameters
    for k in range (0,how_many):
        b.append(lenslist[k][0])
        e.append(lenslist[k][1])
        g.append(lenslist[k][2])
        theta.append(lenslist[k][3])
        rho.append(lenslist[k][4])
        rs.append(lenslist[k][5])

    #loop over each lens and calculate the magnification
    mu = list()
    for i in range (0,how_many):
        #nx = 1600.
        #ny = 1600.
        #ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
        #yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.

        #lpar = n.asarray([b, xcen, ycen, q, P.A.,gamma])
        #fix P.A. of lens to be zero, source is randomly placed so P.A. need not be
        lpar = n.asarray([b[i], 0.0, 0.0, 1-e[i], 0.0, g[i]])
        #gpar = n.asarray([amp., sigma, xcen, ycen, q, P.A.])
        gpar = n.asarray([5.0, rs[i] ,rho[i]*n.cos(theta[i]),rho[i]*n.sin(theta[i]),1.0,0.0])
        xg, yg = gb.sple_grad(ximg,yimg, lpar)
        lmodel = exp_disk(ximg-xg, yimg-yg, gpar)

        #calculate the intrinisic flux (i.e. b=0)
        #nx = 1600.
        #ny = 1600.
        #ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
        #yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.

        smodel = exp_disk(ximg, yimg, gpar)

        #drop the convolved fiber over the images and sum up the flux in the fiber
        f_r = n.sum(lmodel*fiber)
        f_i = n.sum(smodel*fiber)

        mu.append(f_r/f_i)
        smodel=0
        lmodel=0
        ximg=0
        yimg=0

    return mu
    
def lens_show(par, rfib, seeing):
    """
    PURPOSE: Display the lens system in a SDSSIII fiber

    USAGE: lens_show(par, rfib, seeing)

    ARGUMENTS:
    par:
       par[0]: Einstein Radius
       par[1]: ellipticity of lens
       par[2]: power-law index 'gamma': surface density propto R^-gamma
                gamma = 1 is isothermal (r^-2 in 3D);
                gamma > 1 is steeper than isothermal
                gamma < i is shallower than isothermal.
       par[3]: Position angle of source (c.c.w. major-axis rotation w.r.t. x-axis)
       par[4]: radial position of source galaxy in pixels
       par[5]: Effective radius of source galaxy (major axis)
    rfib:  Fiber radius in arcsec
    seeing: FWHM of seeing PSF in arcsec

    RETURNS:  image of source convolved with psf through fiber and image of lensed source
              convolved with psf through fiber

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """
    #
    # Define the pixel scale to be 0.01"/pixel
    #
    #m.use('TkAgg')
    #m.interactive(True)
    myargs = {'interpolation': 'nearest', 'origin': 'lower','cmap': cm.gray, 'hold': False}

    #generate the psf (must be a matching 2d array with lmodel and smodel)
    psf = blur.gauss2d(800,(seeing/0.01))

    nx = 1600.
    ny = 1600.
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    #generate the fiber mask
    fiber = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    for i in range (0,1600):
        for j in range (0,1600):
            fiber[i,j]=n.sqrt(fiber[i,j]**2+(j-800.)**2)

    for i in range (0,1600):
        for j in range (0,1600):
            if fiber[i,j] <= (rfib/0.01):
                fiber[i,j]=1.0
            else: fiber[i,j]=0.0

    #generate the images of source and lensed source
    nx = 1600.
    ny = 1600.
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    #lpar = n.asarray([b, xcen, ycen, q, P.A.,gamma])
    #fix P.A. of lens to be zero, source is randomly placed so P.A. need not be
    lpar = n.asarray([par[0], 0.0, 0.0, 1-par[1], 0.0, par[2]])
    #gpar = n.asarray([amp., sigma, xcen, ycen, q, P.A.])
    gpar = n.asarray([5.0, par[5] ,par[4]*n.cos(par[3]),par[4]*n.sin(par[3]),1.0,0.0])
    xg, yg = gb.sple_grad(ximg,yimg, lpar)
    lmodel = exp_disk(ximg-xg, yimg-yg, gpar)

    #calculate the intrinisic flux (i.e. b=0)
    nx = 1600.
    ny = 1600.
    ximg = n.outer(n.ones(ny), n.arange(nx, dtype='float')) - 800.
    yimg = n.outer(n.arange(ny, dtype='float'), n.ones(nx)) - 800.
    smodel = exp_disk(ximg, yimg, gpar)
    #convolve the models with the psf
    lmodel = signal.fftconvolve(lmodel,psf,mode='same')
    smodel = signal.fftconvolve(smodel,psf,mode='same')
    lmodel = lmodel[1:1601,1:1601]
    smodel = smodel[1:1601,1:1601]
    #display the source and lens models
    p.imshow(n.vstack((n.hstack((lmodel*fiber,smodel*fiber)),n.hstack((lmodel,smodel)))),**myargs)
    #p.imshow(n.hstack((lmodel,smodel)),**myargs)
    #p.imshow(n.hstack((lmodel*fiber,smodel*fiber)),**myargs)
    return
