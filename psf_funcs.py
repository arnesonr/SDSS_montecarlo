import numpy as n
import scipy as s

def gauss2d(size,FWHM):
    """
    PURPOSE: create a 2D gaussian psf for convolutions

    USAGE: psf = gauss2d(size)

    ARGUMENTS:
      size: x & y halfwidth of the gaussian psf (i.e. number of pixels)
      FWHM: Full Width at Half Maximum
      
    RETURNS: 2d gaussian array 
    """
    sigma = FWHM/2.35
    x,y = s.mgrid[-size:size+1,-size:size+1]
    g = n.exp(-(x**2/(2.*sigma**2)+y**2/(2.*sigma**2)))
    
    return g/g.sum()

