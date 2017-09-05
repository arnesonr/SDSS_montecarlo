import numpy as num
import distance_funcs as df
import random as r
from matplotlib import pyplot as p
import random_select_funcs as rs

def pencil_beams(par):
    """
    PURPOSE: Distributes [OII] emission line galaxies within pencil beams
    between 0.0 < z < 1.5 that follow the LF's of Gilbank et al.& Zhu et al.

    USAGE: array = pencil_beams(par[dz, d_f, L_f, L_b, n_beams,z_i,z_f])

    ARGUMENTS:
    pars:
       par[0]: Redshift interval
       par[1]: Fiber diameter (pencil beam diameter) in arcsec.
       par[2]: Faint end luminosity limit
       par[3]: Bright end luminosity limit
       par[4]: Number of pencil beams
       par[5]: Initial Redshift
       par[6]: Final Redshift

    RETURNS:  An array with the beam number, redshift, galaxy luminosity

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """

    #initialize redshift and list of objects
    a=list()
    z = par[5]
    while z <= par[6]:
       #calculate the solid angle of the pencil beam
      # print 'z'
      # print z
       domega = (num.pi*(par[1]/2)**2) * (1.0/206265.0**2)
       #calculate the comoving volume element at the redshift
       dVc = (((3000.0/0.7) * (1.+z)**2)/ num.sqrt(0.3*(1.+z)**3 + 0.7)) * df.Da(z)**2 * par[0] * domega
       #calculate the number density of galaxies at the redshift
       #first calculate the LF parameters
       LF = rs.lum_pars(z) #retruns [alpha_f, alpha_b, L_to, beta_f, beta_b]
       c1 = (10**(-(LF[0]+1)*42.5+LF[3]))/num.log(10)
       c2 = (10**(-(LF[1]+1)*42.5+LF[4]))/num.log(10)
       n = c1*df.trapezoid_rule(lambda L: L**LF[0],par[2],LF[2], 1E6) + c2*df.trapezoid_rule(lambda L:L**LF[1],LF[2],par[3], 1E6)
       #calculate the number of galaxies in the volume element
       N = dVc * n
       #generate a random numbers to compare to N
       u = num.ones(par[4])
       for i in range (0, par[4]):
           u[i] = r.uniform(0,1)
           if u[i] <= N:
              #put a galaxy in this redshift with a randomly selected L and z
              L = rs.light_func([LF[0],LF[1],LF[2],par[2],par[3],1,LF[3],LF[4]])
              z_galaxy = z - r.uniform(0,par[0])
              pencil_beam = i + 1
              #store the redshift, luminosity, and beam # in a list
              a.append([z_galaxy, L, pencil_beam])
              num.save('ELGs',a)
       z = z + par[0]
    return a





