import numpy as n
import mock_lens_funcs as mlf
import random_select_funcs as rsf
import distance_funcs as df

def detect(par):
    """
    PURPOSE: Determine if a randomly selected lens system is detected in the SDSS III pipeline

    USAGE:  array = detect(par)
    
    ARGUMENTS:
       par:
              par[0] Num: integer number of lens galaxies to randomly generate
              par[1] bave: average Einstein radius in arcsec
              par[2] sig_b: sigma of the Einstein radii distribution in arcsec
              par[3] gave: average mass profile parameter
              par[4] sig_g: sigma of the mass profile distribution
              par[5] rfib: radius of the fiber in arcsec
              par[6] seeing: FWHM of seeing psf

    RETURNS:  array of lens and source parameters as well as magnification and
              detection flag (1 if detected, 0 if not detected)

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    
    #make a list of lenses
    #all lenses are assumed to be at a redshift of 0.2
    llist = mlf.lenses([par[0], par[1], par[2], par[3], par[4], par[5]])
    #calculate the magnification of the lenses
    mu = mlf.magnification(llist, par[5], par[6])
    
    #randomly draw a source [OII] luminosity
    
    #source is assumed to be at a redshift of 0.8
    LF = rsf.lum_pars(0.8) #retruns [alpha_f, alpha_b, L_to, beta_f, beta_b]
    #L = light_func(par[alpha_f, alpha_b, L_to, L_f, objs, beta_f, beta_b])
    L = rsf.light_func([LF[0], LF[1], LF[2], 10**39.8, par[0], LF[3], LF[4]]) #returns log([OII])
    #calculate the line flux in the fiber; s = (L*mu)/(4*pi*Dl^2)
    s = list()
    Dl = df.Dl(0.8) #returns distance in MPc
    #convert Dl from Mpc to cm
    Dl = Dl*3.085678E24 
    for i in range (0,par[0]):
        s.append((L[i]*mu[i])/(4.0*n.pi*Dl^2.0))
    
    #decide if s is detected
                      
    #return lenslist, magnification and detection flag
    a = list()
    for i in range (0,par[0]):
        a.append([llist[i][0],llist[i][1],llist[i][2],llist[i][3],llist[i][4], llist[i][5],
              mu[i], s[i]])#, flag[i]])
    
    return a
