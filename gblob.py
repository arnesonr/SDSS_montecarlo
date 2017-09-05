"""
gblob: a lightweight python package for optimizing strong lens
models based upon Gaussian blobs in the source plane.

Copyright 2010 Adam S. Bolton


"""

# Imports:
import numpy as n
import copy
from scipy import signal as sig
from scipy import interpolate as ip
from scipy import sparse
from matplotlib import pyplot as p
from numpy import linalg as la
from scipy import integrate as ig

def make_xy_grid(dimens):
    """
    Function to return x and y pixel coordinate grids.

    Usage:
    ximage, yimage = make_xy_grid((ny, nx))

    Note ordering of ny and nx versus ximage, yimage
"""
    ny, nx = dimens
    ximage = n.outer(n.ones(ny, dtype='float'), n.arange(nx, dtype='float'))
    yimage = n.outer(n.arange(ny, dtype='float'), n.ones(nx, dtype='float'))
    return ximage, yimage


def xy_rotate(x, y, xcen, ycen, phi):
    """
    NAME: xy_rotate

    PURPOSE: Transform input (x, y) coordiantes into the frame of a new
             (x, y) coordinate system that has its origin at the point
             (xcen, ycen) in the old system, and whose x-axis is rotated
             c.c.w. by phi degrees with respect to the original x axis.

    USAGE: (xnew,ynew) = xy_rotate(x, y, xcen, ycen, phi)

    ARGUMENTS:
      x, y: numpy ndarrays with (hopefully) matching sizes
            giving coordinates in the old system
      xcen: old-system x coordinate of the new origin
      ycen: old-system y coordinate of the new origin
      phi: angle c.c.w. in degrees from old x to new x axis

    RETURNS: 2-item tuple containing new x and y coordinate arrays

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    phirad = n.deg2rad(phi)
    xnew = (x - xcen) * n.cos(phirad) + (y - ycen) * n.sin(phirad)
    ynew = (y - ycen) * n.cos(phirad) - (x - xcen) * n.sin(phirad)
    return (xnew,ynew)


def gauss_2d(x, y, fullpar):
    """
    NAME: gauss_2d

    PURPOSE: Implement 2D Gaussian function

    USAGE: z = gauss_2d(x, y, fullpar)

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
      Additional Gaussian components can be added via
      fullpar[6:12], fullpar[12:18], etc.

    RETURNS: 2D Gaussian evaluated at x-y coords

    NOTE: amplitude = 1 is not normalized, but rather has max = 1

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    n_each = 6
    z = 0. * x
    ngauss = len(fullpar) / n_each
    for i in range(ngauss):
        par = fullpar[i*n_each:(i+1)*n_each]
        (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
        r_ell_sq = ((xnew**2)*par[4] + (ynew**2)/par[4]) / n.abs(par[1])**2
        z += par[0] * n.exp(-0.5*r_ell_sq)
    return z

def devauc(x, y, fullpar):
    """
    deVaucouleurs image model function.

    USAGE: z = devauc(x, y, fullpar)

    ARGUMENTS:
      x, y: vecors or images of coordinates;
            should be matching numpy ndarrays
      fullpar: vector of parameters, defined as follows:
        fullpar[0]: surface brightness at the half-light radius
        fullpar[1]: intermediate-axis half-light radius
        fullpar[2]: x-center
        fullpar[3]: y-center
        fullpar[4]: axis ratio
        fullpar[5]: c.c.w. major-axis rotation w.r.t. x-axis
      Additional deVaucouleurs components can be added via
      fullpar[6:12], fullpar[12:18], etc.

    RETURNS: 2D deVaucouleurs evaluated at x-y coords

    WRITTEN: Adam S. Bolton, U. of Utah, 2010
    """
    k_dev = 7.66925001
    n_each=6
    z = 0. * x
    ndev = len(fullpar) / n_each
    for i in range(ndev):
        par = fullpar[i*n_each:(i+1)*n_each]
        (xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
        r_ell = n.sqrt(((xnew**2)*par[4] + (ynew**2)/par[4])) / n.abs(par[1])
        z += par[0] * n.exp(-k_dev * (r_ell**0.25 - 1.))
    return z

def sie_grad(x, y, par):
    """
    NAME: sie_grad

    PURPOSE: compute the deflection of an SIE potential

    USAGE: (xg, yg) = sie_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays
      par: vector of parameters with 1 to 5 elements, defined as follows:
        par[0]: lens strength, or 'Einstein radius'
        par[1]: (optional) x-center (default = 0.0)
        par[2]: (optional) y-center (default = 0.0)
        par[3]: (optional) axis ratio (default=1.0)
        par[4]: (optional) major axis Position Angle
                in degrees c.c.w. of x axis. (default = 0.0)

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)

    NOTES: This routine implements an 'intermediate-axis' convention.
      Analytic forms for the SIE potential can be found in:
        Kassiola & Kovner 1993, ApJ, 417, 450
        Kormann et al. 1994, A&A, 284, 285
        Keeton & Kochanek 1998, ApJ, 495, 157
      The parameter-order convention in this routine differs from that
      of a previous IDL routine of the same name by ASB.

    WRITTEN: Adam S. Bolton, U of Utah, 2009
    """
    # Set parameters:
    b = n.abs(par[0]) # can't be negative!!!
    xzero = 0. if (len(par) < 2) else par[1]
    yzero = 0. if (len(par) < 3) else par[2]
    q = 1. if (len(par) < 4) else n.abs(par[3])
    phiq = 0. if (len(par) < 5) else par[4]
    eps = 0.001 # for sqrt(1/q - q) < eps, a limit expression is used.
    # Handle q > 1 gracefully:
    if (q > 1.):
        q = 1.0 / q
        phiq = phiq + 90.0
    # Go into shifted coordinats of the potential:
    phirad = n.deg2rad(phiq)
    xsie = (x-xzero) * n.cos(phirad) + (y-yzero) * n.sin(phirad)
    ysie = (y-yzero) * n.cos(phirad) - (x-xzero) * n.sin(phirad)
    # Compute potential gradient in the transformed system:
    r_ell = n.sqrt(q * xsie**2 + ysie**2 / q)
    qfact = n.sqrt(1./q - q)
    # (r_ell == 0) terms prevent divide-by-zero problems
    if (qfact >= eps):
        xtg = (b/qfact) * n.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
        ytg = (b/qfact) * n.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
    else:
        xtg = b * xsie / (r_ell + (r_ell == 0))
        ytg = b * ysie / (r_ell + (r_ell == 0))
    # Transform back to un-rotated system:
    xg = xtg * n.cos(phirad) - ytg * n.sin(phirad)
    yg = ytg * n.cos(phirad) + xtg * n.sin(phirad)
    # Return value:
    return (xg, yg)

def sple_grad(x, y, par, isteps=100):
    """
    NAME: sple_grad

    PURPOSE: compute the deflection of an SPLE mass distribution
             (singular power-law ellipsoid)

    USAGE: (xg, yg) = sple_grad(x, y, par)

    ARGUMENTS:
      x, y: vectors or images of coordinates;
            should be matching numpy ndarrays.
            Currently there is no testing for size matching!
      par: vector of parameters with 1 to 6 elements, defined as follows:
        par[0]: lens strength, or 'Einstein radius'
        par[1]: (optional) x-center (default = 0.0)
        par[2]: (optional) y-center (default = 0.0)
        par[3]: (optional) axis ratio (default=1.0)
        par[4]: (optional) major axis Position Angle
                in degrees c.c.w. of x axis. (default = 0.0)
        par[5]: power-law index 'gamma': surface density propto R^-gamma.
                gamma = 1 is isothermal (r^-2 in 3D);
                gamma > 1 is steeper than isothermal
                gamma < i is shallower than isothermal.
      isteps: number of steps in 90deg of azimuth over which to compute,
              for subsequent interpolation.

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)

    NOTES: This routine implements an 'intermediate-axis' convention.
      The parameter-order convention in this routine differs from that
      of a previous IDL routine of the similar name by ASB.

      This routine is not efficient for small numbers (less than ~100)
      of image-plane points.  It is written with the intention of
      working efficiently for evaluation over a large number of image
      plane points during the course of lens-model optimization,
      so it evaluates the expensive integrals over a baseline
      gridded by 'isteps' and interpolates.  If one had fewer than
      'isteps' points in the image plane, it would make more sense just
      to integrate directly for their associated values, but that would
      require additional logic.

      I *think* this is originally based upon Barkana 1998 ApJ, 502, 531,
      with the simplification of zero core radius.  It is translated from
      an IDL code written 6 years ago by teh translator, so memory is hazy.

    WRITTEN: Adam S. Bolton, U of Utah, 2009
    """
    # Extract parameters:
    bpl = n.abs(par[0]) # lens strength, can't be negative!
    xzero = 0. if (len(par) < 2) else par[1]
    yzero = 0. if (len(par) < 3) else par[2]
    q = 1. if (len(par) < 4) else n.abs(par[3])
    phiq = 0. if (len(par) < 5) else par[4]
    gpl = 1. if (len(par) < 6) else par[5]
    # Handle q > 1 gracefully:
    if (q > 1.):
        q = 1.0 / q
        phiq = phiq + 90.0
    # Don't let gpl exceed the bounds [0., 2.]:
    gpl = 2.0 if gpl > 2.0 else gpl
    gpl = 0.0 if gpl < 0.0 else gpl
    # Go into shifted coordinates of the potential:
    phirad = n.deg2rad(phiq)
    xpl = (x-xzero) * n.cos(phirad) + (y-yzero) * n.sin(phirad)
    ypl = (y-yzero) * n.cos(phirad) - (x-xzero) * n.sin(phirad)
    # Store quadrant info and reduce to quadrant 1:
    xnegative = xpl < 0.
    ynegative = ypl < 0.
    xpl = n.abs(xpl)
    ypl = n.abs(ypl)
    # Compute azimuth and radial coordinate.
    # No need to use generalized radial coordinate,
    # since we can just scale it later.
    theta = n.asarray(n.arctan2(ypl, xpl))
    r = n.asarray(n.hypot(xpl, ypl))
    # Compute potential gradient around a ring at unit radius:
    rfid = 1.
    thbase = 0.5 * n.pi * (n.arange(float(isteps+3)) - 1.) / float(isteps)
    xgbase = n.zeros(float(isteps+3))
    ygbase = n.zeros(float(isteps+3))
    for i in range(isteps+3):
        xpl_this = rfid * n.cos(thbase[i])
        ypl_this = rfid * n.sin(thbase[i])
        rhomax = n.sqrt(xpl_this**2 + ypl_this**2 / q**2)
        umax = rhomax**(2. - gpl)
        def pl_integrand_x_2(u):
            rho = u**(1. / (2. - gpl))
            delta = n.sqrt(((1. - q**2) * rho**2 + ypl_this**2 - xpl_this**2)**2
                           + 4. * xpl_this**2 * ypl_this**2)
            omega = n.sqrt((delta + xpl_this**2 + ypl_this**2 + (1. - q**2) * rho**2) /
                           (delta + xpl_this**2 + ypl_this**2 - (1. - q**2) * rho**2))
            integrand = 2. * xpl_this * q * 0.5 * (bpl / n.sqrt(q))**gpl * omega / (xpl_this**2 + omega**4 * ypl_this**2)
            return integrand
        xgbase[i] = ig.romberg(pl_integrand_x_2, 0., umax, vec_func=True)
        def pl_integrand_y_2(u):
            rho = u**(1. / (2. - gpl))
            delta = n.sqrt(((1. - q**2) * rho**2 + ypl_this**2 - xpl_this**2)**2
                           + 4. * xpl_this**2 * ypl_this**2)
            omega = n.sqrt((delta + xpl_this**2 + ypl_this**2 + (1. - q**2) * rho**2) /
                           (delta + xpl_this**2 + ypl_this**2 - (1. - q**2) * rho**2))
            integrand = 2. * ypl_this * q * 0.5 * (bpl / n.sqrt(q))**gpl * omega**3 / (xpl_this**2 + omega**4 * ypl_this**2)
            return integrand
        ygbase[i] = ig.romberg(pl_integrand_y_2, 0., umax, vec_func=True)
    # Compute interpolating splines:
    xgb2func = ip.UnivariateSpline(thbase, xgbase, s=0., k=3)
    ygb2func = ip.UnivariateSpline(thbase, ygbase, s=0., k=3)
    # Evaluate splines for deflection values at r=rfid
    # (gotta flatten, because I think there's some FORTRAN under the hood.)
    xtg = (xgb2func(theta.flatten())).reshape(theta.shape)
    ytg = (ygb2func(theta.flatten())).reshape(theta.shape)
    # Scale to the appropriate values for the actual radii:
    xtg = xtg * ((r + (r == 0)) / rfid)**(1.- gpl) * (r != 0)
    ytg = ytg * ((r + (r == 0)) / rfid)**(1.- gpl) * (r != 0)
    # Restore quadrant-appropriate signs:
    xtg = xtg * (-1.)**xnegative
    ytg = ytg * (-1.)**ynegative
    # Take gradient back into the un-rotated system and return:
    xg = xtg * n.cos(phirad) - ytg * n.sin(phirad)
    yg = ytg * n.cos(phirad) + xtg * n.sin(phirad)
    return (xg, yg)

def shear_grad(x, y, par):
    """
    Constant external shear deflection function
    x, y: vectors or images of coordinates
    par: parameters of the shear mapping:
      par[0]: shear strength
      par[1]: shear position angle (in degrees)
      par[2]: shear coordinate x center
      par[3]: shear coordinate y center
    NOTE: do not attempt to fit for the shear coordinate
      center under normal circumstances, as it will be
      degenerate with the position of your sources.

    RETURNS: tuple (xg, yg) of gradients at the positions (x, y)
    """
    gamma = par[0]
    phig = par[1]
    xc = par[2]
    yc = par[3]
    cosval = n.cos(2. * n.deg2rad(phig))
    sinval = n.sin(2. * n.deg2rad(phig))
    xg = gamma * ((x - xc) * cosval + (y - yc) * sinval)
    yg = gamma * ((x - xc) * sinval - (y - yc) * cosval)
    return (xg, yg)

def create_ltm_grad(kappa, xbase=None, ybase=None):
    """
    Function to take a convergence map and to generate
    a lensing potential gradient function from it, with
    one free scaling parameter.

    Return arrays have deflections in units of pixels.
    """
    # Create centered kernels with which to convolve:
    ny, nx = kappa.shape
    ximage, yimage = make_xy_grid((2*ny+1,2*nx+1))
    ximage -= nx
    yimage -= ny
    xkernel = (1./n.pi) * ximage / ((ximage**2 + yimage**2) + (ximage == 0))
    ykernel = (1./n.pi) * yimage / ((ximage**2 + yimage**2) + (yimage == 0))
    # make a padded version of kappa:
    kappa_pad = n.zeros((3*ny, 3*nx))
    kappa_pad[ny:2*ny,nx:2*nx] = kappa
    # Do convolutions for deflections:
    xdefl = sig.fftconvolve(kappa_pad, xkernel, mode='same')
    ydefl = sig.fftconvolve(kappa_pad, ykernel, mode='same')
    # For debugging, the divergence of the deflection field:
    #div_defl = (xdefl[ny:2*ny,nx+1:2*nx+1] - xdefl[ny:2*ny,nx-1:2*nx-1] +
    #            ydefl[ny+1:2*ny+1,nx:2*nx] - ydefl[ny-1:2*ny-1,nx:2*nx]) / 2.
    xdefl = xdefl[ny:2*ny,nx:2*nx]
    ydefl = ydefl[ny:2*ny,nx:2*nx]
    # Need to make local copies, since they can be changed at the caller level:
    xbase_loc = copy.deepcopy(xbase)
    ybase_loc = copy.deepcopy(ybase)
    def ltm_grad(x, y, par):
        xg_out = par[0] * bilinear(x, y, xdefl, xbase=xbase_loc, ybase=ybase_loc)
        yg_out = par[0] * bilinear(x, y, ydefl, xbase=xbase_loc, ybase=ybase_loc)
        return (xg_out, yg_out)
    return ltm_grad

def create_one2many(lpars=None, spars=None, lfixed=None, sfixed=None):
    """
    Create functions to map from flat to structured parameter
    arguments, with flags for fixed parameters.
    """
    # Have to make local copies of the arguments, otherwise they
    # change within the output function when changed at the caller level.
    # Yuck!  Fun with Python...
    lpars_loc = copy.deepcopy(lpars)
    lfixed_loc = copy.deepcopy(lfixed)
    spars_loc = copy.deepcopy(spars)
    sfixed_loc = copy.deepcopy(sfixed)
    def one2many(pars):
        k = 0
        lpars_out = 0
        spars_out = 0
        # Loop over parameter lists to construct outputs:
        if not (lpars_loc is None):
            lpars_out = copy.deepcopy(lpars_loc)
            for j in range(len(lpars_loc)):
                for i in range(len(lpars_loc[j])):
                    if (lfixed_loc[j][i] == 0):
                        lpars_out[j][i] = pars[k]
                        k +=1
        if not (spars_loc is None):
            spars_out = copy.deepcopy(spars_loc)
            for j in range(len(spars_loc)):
                for i in range(len(spars_loc[j])):
                    if (sfixed_loc[j][i] == 0):
                        spars_out[j][i] = pars[k]
                        k +=1
        return lpars_out, spars_out
    return one2many

def many2one(lpars=None, spars=None, lfixed=None, sfixed=None):
    """
    Functions to map from structured to flat parameter
    arguments, with flags for fixed parameters.
    Unlike the one2many case, no factory function is needed.
    """
    pars = []
    if not (lpars is None):
        for j in range(len(lpars)):
            for i in range(len(lpars[j])):
                if (lfixed[j][i] == 0):
                    pars.append(lpars[j][i])
    if not (spars is None):
        for j in range(len(spars)):
            for i in range(len(spars[j])):
                if (sfixed[j][i] == 0):
                    pars.append(spars[j][i])
    pars = n.asarray(pars)
    return pars

def make9waysubgrid(dimens, origin=(0. ,0.)):
    """
    Function to generate the subsampling grids needed for our quadratic
    sub-pixel integration scheme.
      dimens = (ny, nx)
      origin = (yzero, xzero)
    returns:
      xsubgrid, ysubgrid
    (NOTE ordering conventions!)
    """
    ny, nx = dimens
    y0, x0 = origin
    new_dimens = (2 * ny + 1, 2 * nx + 1)
    xgrid, ygrid = make_xy_grid(new_dimens)
    return 0.5 * xgrid - 0.5 + x0, 0.5 * ygrid - 0.5 + y0

def nine_point_integral(img):
    """
    Function to compute a 9-point integral estimate of an image
    based upon a larger imput image, using the proper weighting
    for biquadratic accuracy.  If the input image has dimension
    2*ny+1, 2*nx+1, the output will have dimension ny, nx.
    Even input dimensions handled reasonably sensibly.
    """
    ny_in, nx_in = img.shape
    # Annoying steps we have to take to make sure that we deal
    # correctly with even input dimensions (we prefer odd):
    ym = ny_in - n.mod(ny_in + 1, 2)
    xm = nx_in - n.mod(nx_in + 1, 2)
    outimg = (16. * img[1:ym-1:2,1:xm-1:2] +   # center
              4. * img[1:ym-1:2,0:xm-2:2] +    # left
              4. * img[1:ym-1:2,2:xm:2] +      # right
              4. * img[0:ym-2:2,1:xm-1:2] +    # bottom
              4. * img[2:ym:2,1:xm-1:2] +      # top
              img[0:ym-2:2,0:xm-2:2] +         # bottom left
              img[0:ym-2:2,2:xm:2] +           # bottom right
              img[2:ym:2,0:xm-2:2] +           # top left
              img[2:ym:2,2:xm:2]               # top right
              ) / 36.                          # normalization
    return outimg

def map2sourceplane(x, y, lfuncs, lpars):
    """
    Convenience function to map the image-plane point sets x and y
    into source plane coordinates using the lens gradient functions
    in the list lfuncs with the parameters specified in lpars.
    """
    xg, yg = 0. * x, 0. * y
    for i in range(len(lfuncs)):
        xg_this, yg_this = lfuncs[i](x, y, lpars[i])
        xg += xg_this
        yg += yg_this
    return x - xg, y - yg

def compute_lensed_image(x, y, lfuncs, sfuncs, lpars, spars):
    """
    Function to compute lensed image given x and y coordinate images,
    lens parameters, source parameters, lens function lists, and source
    function lists.  This function does no PSF integration or subsampling.
    """
    z = 0. * x
    xs, ys = map2sourceplane(x, y, lfuncs, lpars)
    for i in range(len(sfuncs)):
        z += sfuncs[i](xs, ys, spars[i])
    return z

def compute_lensed_pixelimage(lfuncs, lpars, splanepars, data=None, ivar=None,
                              psf=None, full_output=False):
    """
    Function to compute lensed image of a pixelized source plane, given
    lens parameters, lens function lists, and source-plane parameters.

    ARGUMENTS:
    lfuncs: list of lens-mapping functions
    lpars: list of lens-mapping parameter vectors.
      PIXEL-POSITION PARAMETERS MUST BE EXPRESSED IN CENTERED-PIXEL
      IMAGE COORDINATES -- NO X,Y COORDINATE OFFSET ALLOWED!!!
    splanepars: A list or tuple of parameters that describe the source plane.
      SHOULD NOT BE TREATED AS FREE PARAMETERS IN AN OPTIMIZER WRAP!!!
    See make_lensmap_matrix for the context in which these apply.
      splanepars[0]: lowest x coordinate (in mapped pixels) of source plane
      splanepars[1]: lowest y coordinate (in mapped pixels) of source plane
      splanepars[2]: highest x coordinate (in mapped pixels) of source plane
      splanepars[3]: highest y coordinate (in mapped pixels) of source plane
      splanepars[4]: number of x pixels in the source plane
      splanepars[5]: number of y pixels in the source plane
      splanepars[6]: sub-sampling factor of image plane pixels.
        Set this to 1 if no subsampling desired.
    data: lensed image data.
    ivar: inverse variance image of data.
    psf: point-spread function with which to convolve.
    full_output: if true, return a tuple of (modelim, coeffs, icov)
      where coeffs is the unwrapped vector of source plane pixel parameters
      and icov is the inverse covariance matrix of these parameters.
    """
    # Generate the x and y grids:
    ssfact = splanepars[6]
    ny, nx = data.shape
    xsubim, ysubim = xysubgrid(nx=nx, ny=ny, xsub=ssfact, ysub=ssfact)
    # Map these to the source plane:
    xssub, yssub = map2sourceplane(xsubim, ysubim, lfuncs, lpars)
    # Compute the lens-mapping matrix:
    xsrange = n.asarray([splanepars[0], splanepars[2]])
    ysrange = n.asarray([splanepars[1], splanepars[3]])
    xspix = splanepars[4]
    yspix = splanepars[5]
    lmap = make_lensmap_matrix(xsubim, ysubim, xssub, yssub, xsrange, ysrange,
                               xspix, yspix, psf=psf)
    #ldense = lmap.todense()
    # Make sparse diagonal matrix with inverse variance on diag:
    isparse = sparse.lil_matrix((ivar.size, ivar.size), dtype='float32')
    isparse.setdiag(ivar.flatten())
    isparse = isparse.tocsr()
    # Generate inverse covariance matrix of source plane parameters:
    ilmap = lmap.transpose() * isparse
    icov = ilmap * lmap
    # Multiply for "right-hand side":
    beta = ilmap * data.flatten()
    # Solve in dense form, for now:
    coeffs = la.solve(icov.todense(), beta)
    modelim = (lmap * coeffs).reshape((ny, nx))
    if full_output:
        return modelim, coeffs, icov
    else:
        return modelim

def create_lensoptfunc(data=None, ivar=None, x0=0., y0=0.,
                       lfuncs=None, sfuncs=None, lpars=None, spars=None,
                       psf=None, subsamp=True, lfixed=None, sfixed=None, plot=False,
                       pixelized=False):
    """
    A function to take many elements of a multi-component lens model
    and corresponding source model viewed through it,
    and return a single function of a single parameter vector that
    may be optimized by scipy.optimize.leastsq

    Can also be pressed into service as an image modeler without any
    lens model in front.
    """
    # Create the "one2many" function needed for argument parsing
    # by the function to be created:
    one2many = create_one2many(lpars, spars, lfixed, sfixed)
    # Create the necessary coordinate grids:
    if subsamp:
        x, y = make9waysubgrid(data.shape, (y0, x0))
    else:
        x, y = make_xy_grid(data.shape)
        x += x0
        y += y0
    # We have to make local copies of the arguments, otherwise
    # the manufactured function can change when the arguments
    # change at the caller level:
    ivar_loc = ivar.copy()
    data_loc = data.copy()
    psf_loc = psf.copy()
    lfuncs_loc = copy.deepcopy(lfuncs)
    sfuncs_loc = copy.deepcopy(sfuncs)
    # Apply PSF mask to inverse variance if necessary:
    if not (psf is None):
        yw, xw = psf_loc.shape
        xhw = xw / 2
        yhw = yw / 2
        ivar_loc[:,:xhw] = 0.
        ivar_loc[:,-xhw:] = 0.
        ivar_loc[:yhw,:] = 0.
        ivar_loc[-yhw:,:] = 0.
    # Now define the lens optimization function:
    def lensoptfunc(pars, return_model=False, use_psf=True):
        # Parse parameter vector into lens and source parameters in the local scope:
        lpars_loc, spars_loc = one2many(pars)
        # Compute lensed image:
        img = compute_lensed_image(x, y, lfuncs_loc, sfuncs_loc, lpars_loc, spars_loc)
        # Integrate if necessary:
        if subsamp:
            img = nine_point_integral(img)
        # PSF convolve if necessary:
        if ((use_psf) and (not (psf_loc is None))):
            img = sig.fftconvolve(img, psf_loc, mode='same')
        if plot:
            p.imshow(n.hstack((data_loc, img, data_loc-img)), hold=False, vmin=data_loc.min(), vmax=data_loc.max())
        if not return_model:
            #img = ((data_loc - img)**2 * ivar_loc).flatten()
            img = ((data_loc - img) * n.sqrt(ivar_loc)).flatten()
        return img
    return lensoptfunc

def create_pixel_lensoptfunc(data=None, ivar=None,
                             lfuncs=None, lpars=None, splanepars=None,
                             psf=None, lfixed=None, plot=False):
    """
    A function to take many elements of a multi-component lens model
    and specificed pixelized source plane,
    and return a single function of a single parameter vector that
    may be optimized by scipy.optimize.leastsq.
    See compute_lensed_pixelimage for more argument documentation.
    """
    # Create the "one2many" function needed for argument parsing
    # by the function to be created (source plane parameters
    # not meant to be optimized, so they are fixed)
    #sfixed = [n.asarray([1]) for i in splanepars]
    #spars = [n.asarray([x]) for x in splanepars]
    one2many = create_one2many(lpars=lpars, lfixed=lfixed) #, sfixed)
    # We have to make local copies of the arguments, otherwise
    # the manufactured function can change when the arguments
    # change at the caller level:
    ivar_loc = ivar.copy()
    data_loc = data.copy()
    psf_loc = psf.copy()
    lfuncs_loc = copy.deepcopy(lfuncs)
    splanepars_loc = copy.deepcopy(splanepars)
    # Now define the lens optimization function:
    def lensoptfunc(pars, return_model=False):
        # Parse parameter vector into lens and source parameters in the local scope:
        lpars_loc, spars_loc = one2many(pars)
        # Compute lensed image:
        img = compute_lensed_pixelimage(lfuncs_loc, lpars_loc, splanepars_loc,
                                        data=data_loc, ivar=ivar_loc, psf=psf_loc)
        if plot:
            p.imshow(n.hstack((data_loc, img, data_loc-img)), hold=False, vmin=data_loc.min(), vmax=data_loc.max())
        if not return_model:
            #img = ((data_loc - img)**2 * ivar_loc).flatten()
            img = ((data_loc - img) * n.sqrt(ivar_loc)).flatten()
        return img
    return lensoptfunc

def xysubgrid(nx, ny, xsub, ysub):
    """
    Function to return uniformly subsampled x and y coordinate grids

    Arguments:
      nx, ny: unit dimensions of grid
      xsub, ysub: sub-sampling factors in x and y
    ALL ARGUMENTS SHOULD BE INTEGERS GREATER THAN ZERO!!!

    Returns:
      ximg, yimg: tuple of two subsampled coordinate images,
        of dimension ny*ysub X nx*xsub

    NOTES:
      Observes convention whereby pixel coordinates are aligned
      with pixel CENTERS, not edges.
      Observes zero-based indexing convention.
    """
    xbase = 0.5 * (2. * n.arange(nx * xsub) + 1. - xsub) / float(xsub)
    ybase = 0.5 * (2. * n.arange(ny * ysub) + 1. - ysub) / float(ysub)
    xim = n.outer(n.ones(ny*ysub), xbase)
    yim = n.outer(ybase, n.ones(nx*xsub))
    return xim, yim

def make_lensmap_matrix(ximg, yimg, xsrc, ysrc, xsrange, ysrange, xspix, yspix, psf=None):
    """
    Function to generate a linear lens mapping matrix from source plane
    pixel brightnesses to image plane brightnesses.

    Arguments:
    ximg, yimg: matching pixel coordinate arrays in the image plane.
      These may be sub-sampled, and should uniformly smple the image plane.
      Should probably be contiguous and 2D, but may not need to be
      (no guarantees on that!).  Routine observes pixel-center coordinate
      convention.  Image plane is assumed to be of size nx = max(round(ximg))
      by ny = max(round(yimage)); any input coordinates that round to less than
      zero are ignored.  Only the rounded versions of these arrays are actually
      used, so it wouldn't hurt for them to be rounded upon input, but it's
      probably more convenient to pass in the un-rounded version that you
      must have had to create in order to compute the lens mapping.
    xsrc, ysrc: Pixel coordinates in the image plane onto which ximg, yimg map
      via the lens mapping under consideration.  Same length scale as ximg, yimg.
    xsrange, ysrange: 2-element arrays giving the lower (element 0) and upper
      (element 1) boundaries of the pixelized model source plane, in the same
      pixel units as xsrc, ysrc.
    xspix, yspix: number of pixels into which to divide the model source plane
      in the x and y dimensions.  Note that we use a pixel-centered coordinate
      convention, so e.g., xsrange[1] - xsrange[0] will be divided into
      xspix - 1 intervals.
    psf: point-spread function by which to convolve (defaults to no convolution).
      If PSF dimensions are not odd, behavior may be wierd, since this routine
      centers the PSF over the data.

    Returns:
      A scipy sparse matrix of dimension (nx*ny) X (xspix*yspix) that linearly
      maps source plane pixel brightnesses to image plane pixel brightnesses.
      image-plane index wraps as nx*y + x, and source-plane index wraps as
      xspix*y + x.

    Written by Adam S. Bolton, University of Utah, 2010
    """
    # Round image plane pixel coords and determine image plane size:
    xiround = n.int32(n.round(ximg))
    yiround = n.int32(n.round(yimg))
    nx = xiround.max() + 1
    ny = yiround.max() + 1
    # Find the source-plane pixels associated with the mapped points:
    dxsource = (xsrange[1] - xsrange[0]) / float(xspix - 1)
    dysource = (ysrange[1] - ysrange[0]) / float(yspix - 1)
    xsround = n.int32(n.round((xsrc - xsrange[0]) / dxsource))
    ysround = n.int32(n.round((ysrc - ysrange[0]) / dysource))
    # Flatten the rounded index arrays:
    xiround = xiround.flatten()
    yiround = yiround.flatten()
    xsround = xsround.flatten()
    ysround = ysround.flatten()
    # Cut down to samples within the specified source plane range:
    wh = (n.where((xsround >= 0) * (xsround < xspix) *
                  (ysround >= 0) * (ysround < yspix)))[0]
    xiround = xiround[wh]
    yiround = yiround[wh]
    xsround = xsround[wh]
    ysround = ysround[wh]
    ncontrib = wh.size
    wh = 0
    # Make a dummy PSF if one was not supplied:
    if (psf is None):
        psf = n.ones((1,1), dtype='float32')
    # Determine PSF sizes and half-widths:
    ypsf, xpsf = psf.shape
    xhw = xpsf / 2
    yhw = ypsf / 2
    # Make a local copy of the PSF to reshape for broadcasting:
    psf_loc = (psf.reshape((1, ypsf, xpsf))).copy()
    psf_full = (n.ones((ncontrib, 1, 1), dtype='float32') * psf_loc).reshape(ncontrib*ypsf*xpsf)
    xiround = xiround.reshape((ncontrib, 1, 1))
    yiround = yiround.reshape((ncontrib, 1, 1))
    xsround = xsround.reshape((ncontrib, 1, 1))
    ysround = ysround.reshape((ncontrib, 1, 1))
    xoffset = (n.outer(n.ones(ypsf, dtype='int32'), (n.arange(xpsf, dtype='int32') - xhw))).reshape((1,ypsf,xpsf))
    yoffset = (n.outer((n.arange(ypsf, dtype='int32') - yhw), n.ones(xpsf, dtype='int32'))).reshape((1,ypsf,xpsf))
    cdummy = n.zeros((1, ypsf, xpsf), dtype='int32')
    xifull = ((xiround + xoffset).reshape(ncontrib*ypsf*xpsf)).copy()
    xiround = 0
    yifull = ((yiround + yoffset).reshape(ncontrib*ypsf*xpsf)).copy()
    yiround = 0
    xsfull = ((xsround + cdummy).reshape(ncontrib*ypsf*xpsf)).copy()
    xsround = 0
    ysfull = ((ysround + cdummy).reshape(ncontrib*ypsf*xpsf)).copy()
    ysround = 0
    inplane = (n.where((xifull >= 0) * (xifull < nx) * (yifull >= 0) * (yifull < ny)))[0]
    n_in = inplane.size
    i_id = yifull[inplane] * nx + xifull[inplane]
    xifull = yifull = 0
    s_id = ysfull[inplane] * xspix + xsfull[inplane]
    xsfull = ysfull = 0
    psf_full = psf_full[inplane]
    inplane = 0
    lensmap = sparse.coo_matrix((psf_full, n.vstack((i_id,s_id))),shape=(nx*ny,xspix*yspix))
    psf_full = i_id = s_id = 0
    lensmap = lensmap.tocsr()
# Method commented out below is original implementation, about 2.3 x slower
# in the limited test cases considered.  Maybe a bit more efficient with
# memory usage, but much slower in terms of math.
#    # Initialize the output matrix:
#    lensmap = sparse.csr_matrix((ny * nx, yspix * xspix), dtype='float32')
#    # Loop over PSF elements to accumulate the matrix elements:
#    for j in range(ypsf):
#        for i in range(xpsf):
#            # PSF-based offsets in the image plane:
#            xoffset = i - xhw
#            yoffset = j - yhw
#            # Figure out which contributions are still in the image plane:
#            inplane = (n.where( ((xiround + xoffset) >= 0) *
#                                ((xiround + xoffset) < nx) *
#                                ((yiround + yoffset) >= 0) *
#                                ((yiround + yoffset) < ny)))[0]
#            # Construct the input arrays for matrix creation from coordinates
#            # (values to contribute, plus 1D wrapped image- and source-plane indices):
#            nthis = inplane.size
#            valu = psf[j,i] * n.ones(nthis, dtype='float32')
#            i_id = (yiround[inplane] + yoffset) * nx + (xiround[inplane] + xoffset)
#            s_id = ysround[inplane] * xspix + xsround[inplane]
#            # Create output matrix contribution for this PSF pixel and accumulate:
#            lmap_this = sparse.coo_matrix((valu, n.vstack((i_id,s_id))),shape=(nx*ny,xspix*yspix))
#            lensmap = lensmap + lmap_this.tocsr()
    return lensmap

def sym_sqrt(a):
    """
    NAME: sym_sqrt

    PURPOSE: take 'square root' of a symmetric matrix via diagonalization

    USAGE: s = sym_sqrt(a)

    ARGUMENT: a: real symmetric square 2D ndarray

    RETURNS: s such that a = numpy.dot(s, s)

    WRITTEN: Adam S. Bolton, U. of Utah, 2009
    """
    w, v = n.linalg.eigh(a)
    dm = n.diagflat(n.sqrt(w))
    return n.dot(v, n.dot(dm, n.transpose(v)))

def resolution_from_icov(icov):
    """
    Function to generate the 'resolution matrix' in the simplest
    (no unrelated crosstalk) Bolton & Schlegel 2010 sense.
    Works on dense matrices.  Not suited for production-scale
    determination in a spectro extraction pipeline.

    Input argument is inverse covariance matrix array.
    If input is not 2D and symmetric, results will be unpredictable.
    """
    sqrt_icov = sym_sqrt(icov)
    norm_vector = n.sum(sqrt_icov, axis=1)
    r_mat = n.outer(norm_vector**(-1), n.ones(norm_vector.size)) * sqrt_icov
    return r_mat

def bilinear(xinterp, yinterp, zbase, xbase=None, ybase=None):
    """
    Bilinear interpolation function.

    Arguments:
    xinterp, yinterp:
      x and y coordinates at which to interpolate
      (numpy ndarrays of any matching shape)
    zbase:
      Two-dimensional grid of values within which to interpolate.
    xbase, ybase:
      Baseline coordinates of the zbase image.
      should be monotonically increasing numpy 1D ndarrays, with
      lengths matched to the shape of zbase.
      If not set, these default to n.arange(nx) and n.arange(ny),
      where ny, nx = zbase.shape

    Returns:
      An array of interpolated values, with the same shape as xinterp.

    Written: Adam S. Bolton, U of Utah, 2010apr
    """
    ny, nx = zbase.shape
    if (xbase is None):
        xbase = n.arange(nx, dtype='float')
    if (ybase is None):
        ybase = n.arange(ny, dtype='float')
    # The digitize bins give the indices of the *upper* interpolating points:
    xhi = n.digitize(xinterp.flatten(), xbase)
    yhi = n.digitize(yinterp.flatten(), ybase)
    # Reassign to the highest bin if out of bounds above, and to the
    # lowest bin if out of bounds below, relying on the conventions
    # of numpy.digitize to return nx & ny or zero in these cases:
    xhi = xhi - (xhi == nx) + (xhi == 0)
    yhi = yhi - (yhi == ny) + (yhi == 0)
    # The indices of the *lower* bins:
    xlo = xhi - 1
    ylo = yhi - 1
    # The fractional positions within the bins:
    fracx = (xinterp.flatten() - xbase[xlo]) / (xbase[xhi] - xbase[xlo])
    fracy = (yinterp.flatten() - ybase[ylo]) / (ybase[yhi] - ybase[ylo])
    # The interpolation formula:
    zinterp = (zbase[ylo,xlo] * (1. - fracx) * (1. - fracy) +
               zbase[ylo,xhi] * fracx * (1. - fracy) +
               zbase[yhi,xlo] * (1. - fracx) * fracy +
               zbase[yhi,xhi] * fracx * fracy).reshape(xinterp.shape)
    return zinterp




