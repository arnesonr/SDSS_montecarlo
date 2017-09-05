import numpy as N

def trapezoid_rule(f,a,b,n):
    """
    Purpose: Approximates the integral of f from a to b using the trapezoid
             method with n intervals.

    Usage: Area = trapezoid_rule(lambda x: f(x), a, b, n)

    Arguments:
       f: lambda function to be integrated
       a: lower limit of integration
       b: upper limit of integration
       n: number of intervals

    Returns: Area under the curve f(x)

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    return (b-a) * (f(a)/2. + f(b)/2. +N.sum([f(a+(b-a)*k/n) for k in xrange (1,n)])) / n


def Da_distance_graph(f,a,b,n):
    """
    Purpose: Computes the angular diameter distance

    Usage: [Da/DH (z)]= Da_distance_graph(lambda z:f(z), a, b, n)

    Arguments:
       f: lambda function of the transverse comoving distance
       a: lower limit of integration (i.e. lower z)
       b: upper limit of integration (i.e. higer z)
       n: number of intervals between a and b

    Returns: Array from a to b (with n elements) of dimensionless angular
             diameter distance (Da/DH)

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    x = N.arange(a,b,(b-a)/n)

    for i in range(0,N.size(x)):
        x[i] = trapezoid_rule(f,0.,x[i],1+i)/(1+x[i])

    return x

def Da(z):
    """
    Purpose: Compute the angular diameter distance of an object
             at a redshift of z in MPc, assuming standard cosmology
             (i.e. omega_m = 0.3, omega_Lambda = 0.7,h = 0.7)  

    Usage: Da=Da(z)

    Arguments:
       z: redshift of object

    Returns: Angular diameter distance in MPc

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    #Compute the Hubble distance in MPc using h = 0.7
    DH = (3000./.7)
    
    a = (trapezoid_rule(lambda z: (1./N.sqrt(.3*(1.+z)**3.+.7)), 0.0, z, 100000)*(DH))/(1+z)

    return a

def Dm(z):
    """
    Purpose: Compute the comoving distance (equivalent to the transverse comoving distance
             in this cosmology also the current proper distance dp(to))
             of an object at a redshift of z in MPc, assuming standard cosmology
             (i.e. omega_m = 0.3, omega_Lambda = 0.7,h = 0.7)  

    Usage: Dm=Dm(z)

    Arguments:
       z: redshift of object

    Returns: Transverse comoving distance (current proper distance) in MPc

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    #Compute the Hubble distance in MPc using h = 0.7
    DH = (3000/.7)

    m = (trapezoid_rule(lambda z: (1/N.sqrt(.3*(1+z)**3+.7)), 0.0, z, 100000)*(DH))

    return m

def Dl(z):
    """
    Purpose: Compute the luminosity distance of an object at a redshift of z in MPc
             assuming standard cosmology (i.e. omega_m = 0.3, omega_Lambda = 0.7,h = 0.7)  

    Usage: Dl=Dl(z)

    Arguments:
       z: redshift of object

    Returns: Luminosity distance in MPc

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    #Compute the Hubble distance in MPc using h = 0.7
    DH = (3000/.7)

    l = (trapezoid_rule(lambda z: (1./N.sqrt(.3*(1.+z)**3.+.7)), 0.0, z, 100000)*(DH))*(1.+z)

    return l

def Dls(z_l, z_s):
    """
    Purpose: Compute the angular diameter distance between the lens and source in MPc.
             Or between two objects where z_s > z_l, assuming standard cosmology
             (i.e. omega_m = 0.3, omega_Lambda = 0.7,h = 0.7)  

    Usage: Dls=Dls(z_l, z_s)

    Arguments:
       z_l: redshift of the lens
       z_s: redshift of the source

    Returns: Angular diameter distacne in MPc between lens and source

    Written: Ryan A. Arneson, U. of Utah, 2010
    """
    ls = (1./(1.+z_s)) * (Dm(z_s) - Dm(z_l))

    return ls




    
    
