import numpy as num
import distance_funcs as df
import numpy.random as r
from matplotlib import pyplot as p
import random_select_funcs as rs

def pencil_beams(par):
    """
    PURPOSE: Distributes [OII] emission line galaxies within pencil beams
    between 0.0 < z < 1.5 that follow the LF's of Gilbank et al.& Zhu et al.

    USAGE: array = pencil_beams(par[dz, d_f, L_f, n_beams, z_i, z_f, filename])

    ARGUMENTS:
    pars:
       par[0]: Redshift interval
       par[1]: Fiber diameter (pencil beam diameter) in arcsec.
       par[2]: Faint end luminosity limit
       par[3]: Number of pencil beams
       par[4]: Initial Redshift
       par[5]: Final Redshift
       par[6]: Filename to write list to

    RETURNS:  A list with the beam number, redshift, galaxy luminosity

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """

    #initialize redshift and list of objects
    a = list()
    z = par[4]
    #merge all the beams together:
    #solid angle of one beam
    domega = (num.pi*(par[1]/2)**2) * (1.0/206265.0**2)
    #solid angle of all the beams
    A = domega * par[3]
    while z <= (par[5]+(.1*par[0])):
       print 'z'
       print z
       #calculate the comoving volume element at the redshift
       dVc = (((3000.0/0.7) * (1.+z)**2)/ num.sqrt(0.3*(1.+z)**3 + 0.7)) * df.Da(z)**2 * par[0] * A
       #calculate the number density of galaxies at the redshift
       #first calculate the LF parameters
       LF = rs.lum_pars(z) #retruns [alpha_f, alpha_b, L_to, beta_f, beta_b]
       c1 = (10**(-(LF[0]+1)*42.5+LF[3]))/num.log(10)
       c2 = (10**(-(LF[1]+1)*42.5+LF[4]))/num.log(10)
       a1 = LF[0]+1
       a2 = LF[1]+1
       n = c1*(LF[2]**a1/a1-par[2]**a1/a1) + c2*(-LF[2]**a2/a2)
       #calculate the number of galaxies in the volume element
       N = dVc * n
       #decide how many galaxies to put in the slice using a Poisson distribution
       Nthis = r.poisson(lam=N)
       #put Nthis galaxies in this redshift with a randomly selected L, z, and pencil_beam
       for i in range (0,Nthis):
           L = rs.light_func([LF[0],LF[1],LF[2],par[2],1,LF[3],LF[4]])
           z_galaxy = z - r.uniform(0,par[0])
           pencil_beam = r.randint(0,par[3])
           #store the redshift, luminosity, and beam # in a list
           a.append([z_galaxy, L, pencil_beam])
       z = z + par[0]
    num.save(par[6],a)
    return a
