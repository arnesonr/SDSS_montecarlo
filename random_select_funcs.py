import numpy as n
import distance_funcs as df
import random as r
from matplotlib import pyplot as p

def light_func(par):
    """
    PURPOSE: Randomly draws dn/dL's (phi) that follow a double power law

    USAGE: array = light_func(par[alpha_f, alpha_b, L_to, L_f, objs, beta_f, beta_b])

    ARGUMENTS:
    pars:
       par[0]: alpha faint (power of the faint end luminoisty)
       par[1]: alpha bright (power of the bright end luminosity)
       par[2]: Turnover luminosity value of the double power law
       par[3]: faint end luminosity limit
       par[4]: number of objects to select from the light function
       par[5]: beta faint
       par[6]: beta bright

    RETURNS:  An array (or single value) of n objects with values of log L that correspond
              to the appropriate light function.

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """
    #calculate the normalization constant
    c1 = (10**(-(par[0]+1)*42.5+par[5]))/n.log(10)
    c2 = (10**(-(par[1]+1)*42.5+par[6]))/n.log(10)
    A = 1.0/ (c1*(((par[2]**(par[0]+1))/(par[0]+1)) - ((par[3]**(par[0]+1))/(par[0]+1))) - ((c2*par[2]**(par[1]+1))/(par[1]+1)))
    #calculate the cutoff value
    cutoff_value = A * c1 * ((par[2]**(par[0]+1)/(par[0]+1))-(par[3]**(par[0]+1)/(par[0]+1)))
    u = n.ones(par[4])
    if par[4] == 1:
        x_chosen = 1.0
    elif par[4] > 1:
        x_chosen = n.ones(par[4])

    for i in range (0,par[4]):
        u[i]=r.uniform(0,1)
        if u[i] <= cutoff_value:
            c = par[0]+1.0
            if par[4] ==1:
                x_chosen = n.log10((((c*u[i])/(c1*A)) + par[3]**c)**(1.0/c))
            elif par[4] > 1:
                x_chosen[i] = n.log10((((c*u[i])/(c1*A)) + par[3]**c)**(1.0/c))
        if u[i] > cutoff_value:
            c = (par[1]+1.0)
            if par[4]==1.0:
                x_chosen = n.log10((((c*(u[i]-cutoff_value))/(c2*A)) + par[2]**c)**(1.0/c))
            elif par[4]>1:
                x_chosen[i] = n.log10((((c*(u[i]-cutoff_value))/(c2*A)) + par[2]**c)**(1.0/c))
    return x_chosen
                                  
def lum_pars(z):
    """
    PURPOSE: computes the [OII] luminosity parameters:
                   alpha_f: faint end luminosity power
                   alpha_b: bright end luminosity power
                   L_to: interpolated from Gilbank et al. and Zhu et al.
                              using a linear fit (see linear_fits.py)
                   beta_f: normalization parameter for faint end luminosities
                   beta_b: normalization parameter for faint end luminosities
    USAGE: lum_pars = lum_pars(z)

    ARGUMENTS:
       z: redshift at which to compute [OII] luminosity parameters

    RETURNS:  lum_pars
                lum_pars[0]: alpha_f
                lum_pars[1]: alpha_b
                lum_pars[2]: L_to
                lum_pars[3]: beta_f
                lum_pars[4]: beta_b

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2010
    """
    # the linear fit for alpha_f is: alpha_f = m2 * log(1+z) + b2; where:
    m2 = 1.347635
    b2 = -1.734234
    alpha_f = m2 * n.log10(1+z) + b2

    # the linear fit for alpha_b is: alpha_b = m3 * log(1+z) + b3; where:
    m3 = -0.437071
    b3 = -2.809167
    alpha_b = m3 * n.log10(1+z) + b3

    # the linear fit for L_to is: log(L_to) = m * log(1+z) + b; where:
    m = 3.000336
    b = 40.882663
    L_to = 10**(m * n.log10(1+z) + b)

    # the linear fit for beta bright is: beta_b = m4 * log(1+z) + b4; where:
    m4 = 5.454370
    b4 = -5.503379
    beta_b = (m4 * n.log10(1+z) + b4)

    # the linear fit for beta faint is: beta_f = m5 * log(1+z) + b5; where:
    m5 = 3.135254
    b5 = -3.701839
    beta_f = (m5 * n.log10(1+z) + b5)
    
    pars = [alpha_f, alpha_b, L_to, beta_f, beta_b]
    
    return pars
