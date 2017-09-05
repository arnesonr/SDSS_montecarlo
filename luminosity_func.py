import numpy as n
import distance_funcs as df
import random as r
from matplotlib import pyplot as p
import random_select_funcs as rs

#compute the LF
lum_pars = [-1.67, -2.83, 10**41.03, 10**39.8, 10**45.0, 10000]
x = rs.light_func(lum_pars)
p.hist(x, 50, log='True')

#see how actual LF compares with interpolated LF parameters
pars = rs.lum_pars(.116)
y = rs.light_func(pars+[10**39.8, 10**45.0, 10000])
p.hist(y, 50, log='True')

################################################################


#compute the LF at each redshift
pars = rs.lum_pars(.05)
phi = rs.light_func([pars[0],pars[1],pars[2],10**39.8,1.E6,pars[3],pars[4]])
LF = p.hist(phi, 50, log='True') #LF[0]=n; LF[1]=L
#calculate dL (d log L)
dL = LF[1][1]-LF[1][0]
#go through LF to see if there is a galaxy with a given Luminosity
N = n.zeros(50)
for i in range (0,50):
    N[i] = LF[0][i] * dL * V


lum_pars = [10**-2.8, -1.67, -2.83, 10**41.03, 10**39.8, 10**45.0]

#compute the total number density of galaxies
dn = df.trapezoid_rule(lambda L: lum_pars[0]*(L/lum_pars[3])**lum_pars[1], lum_pars[4], lum_pars[3], 100000.0) + df.trapezoid_rule(lambda L: lum_pars[0]*(L/lum_pars[3])**lum_pars[2], lum_pars[3], lum_pars[5], 100000.0)

z = 0.0
d_f = 10.0 #fiber diameter in arcsec.

phi = n.arange(0.0,1.51,.01)
z=0.0
dz = 0.01
for i in range (0,151):
        #calculate the solid angle of the pencil beam
        domega = (n.pi*(d_f/2)**2) * (1.0/206265.0**2)
        #calculate the comoving volume element at each redshift
        dVc = (((3000/0.7) * (1+z)**2 )/ n.sqrt(0.3*(1+z)**3 + 0.7)) * df.Da(z)**2 * dz * domega
        
        if z < 0.75:
            phi[i] = dVc*dn
        
        if 0.75 <= z < 0.93:
            phi[i] = dVc*dn_1
        
        if 0.93 <= z < 1.10:
            phi[i] = dVc*dn_2
        
        if 1.10 <= z < 1.28:
            phi[i] = dVc*dn_3
        
        elif z < 1.6:
            phi[i] = dVc*dn_4
        
        z = z + dz


########################################################################
alpha = -1.67
Lto = 10**41.03

c = 10**-2.57/(Lto**(alpha+1)*n.log(10))
dn_f = c * df.trapezoid_rule(lambda L: L**(alpha), 10**39.8, Lto, 1000000)

alpha = -2.83
c = 10**-2.57/(Lto**(alpha+1)*n.log(10))
dn_b = c * df.trapezoid_rule(lambda L: L**(alpha), Lto, 10**45.0,  1000000)
dn = dn_f + dn_b
#====================
alpha = -1.3
beta = -4.01
Lto = 10**41.68

c = 10**(-(alpha+1)*42.5 + beta)
dn_f1 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, 10**39.8, Lto, 1000000)

alpha = -3.01
c = 10**(-(alpha+1)*42.5 + beta)
dn_b1 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, Lto, 10**45.0, 1000000)
dn_1 = dn_f1 +dn_b1
#=====================
alpha = -1.3
beta = -3.86
Lto = 10**41.75

c = 10**(-(alpha+1)*42.5 + beta)
dn_f2 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, 10**39.8, Lto, 1000000)

alpha = -2.98
c = 10**(-(alpha+1)*42.5 + beta)
dn_b2 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, Lto, 10**45.0, 1000000)
dn_2 = dn_f2 +dn_b2
#=====================
alpha = -1.3
beta = -3.67
Lto = 10**41.93

c = 10**(-(alpha+1)*42.5 + beta)
dn_f3 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, 10**39.8, Lto, 1000000)

alpha = -3.03
c = 10**(-(alpha+1)*42.5 + beta)
dn_b3 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, Lto, 10**45.0, 1000000)
dn_3 = dn_f3 +dn_b3
#=====================
alpha = -1.3
beta = -3.49
Lto = 10**42.0

c = 10**(-(alpha+1)*42.5 + beta)
dn_f4 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, 10**39.8, Lto, 1000000)

alpha = -2.79
c = 10**(-(alpha+1)*42.5 + beta)
dn_b4 = (c/n.log(10)) * df.trapezoid_rule(lambda L: L**alpha, Lto, 10**45.0, 1000000)
dn_4 = dn_f4 +dn_b4
#=====================
#
# 
#
Lto = 41.03
L = n.arange(39.8, 42.0, .01)
phi_new = n.zeros(n.size(L))
phi_old = n.zeros(n.size(L))
for i in range (0,n.size(L)):
    if L[i] < Lto:
        phi_new[i] = n.log10(10**-2.57*(10**L[i]/10**Lto)**-0.67)
        phi_old[i] = n.log10(10**-2.57*(10**L[i]/10**Lto)**-1.67)
    if L[i] > Lto:
        phi_new[i] = n.log10(10**-2.57*(10**L[i]/10**Lto)**-1.83)
        phi_old[i] = n.log10(10**-2.57*(10**L[i]/10**Lto)**-2.83)
########
#
#
##The following is to generate a graph of the data vs. rs.light_func program
#
l = n.arange(39.8,42.4,.2)
phi_plot =n.array([14.031,13.292,9.922,8.141,6.23,4.343,2.707,1.393,.622,.243,.083,.08,.024,.016])
phi_plot = n.log10(phi_plot*10**-3)
p.plot(l,phi_plot,'*',label='data')
p.plot(L,phi_new,label='phi_new')

L = n.arange(39.8,43.0, .01)
phi_gil = n.zeros(n.size(L))
pars = rs.lum_pars(.116)
for i in range (0,n.size(L)):
    if L[i] <= n.log10(pars[2]):
        phi_gil[i] = ((pars[0]+1)*(L[i]-42.5)+pars[3])
    if L[i] >= n.log10(pars[2]):
        phi_gil[i] = ((pars[1]+1)*(L[i]-42.5)+pars[4])

p.plot(l,phi_plot,'o',color='r')
p.plot(L,phi_gil,color='r')
##
l = n.arange(41.15, 43.55, .15)
z1_plot = n.array([49.49, 60.57, 56.07, 46.93, 29.0, 17.68, 7.91, 4.7, 1.62, 1.1, .18, .18, .27, .09, 1000, .04])
z1_plot = n.log10(z1_plot*10**-4)
z1_pars = rs.lum_pars(.837)
L = n.arange(39.8,43.55, .01)
phi_z1 = n.zeros(n.size(L))
for i in range (0,n.size(L)):
    if L[i] <= n.log10(z1_pars[2]):
        phi_z1[i] = ((z1_pars[0]+1)*(L[i]-42.5)+z1_pars[3])
    if L[i] >= n.log10(z1_pars[2]):
        phi_z1[i] = ((z1_pars[1]+1)*(L[i]-42.5)+z1_pars[4])

p.plot(l,z1_plot,'o',color='y')
p.plot(L, phi_z1,color='y')
##
l = n.arange(41.3, 43.85, .15)
z2_plot = n.array([20.69, 28.25, 35.57, 31.3, 20.38, 11.6, 6.14, 2.77, 1.13, .78, .27, .07, .15, .06, 1000, .03, 1000, .03])
z2_plot = n.log10(z2_plot*10**-4)
z2_pars = rs.lum_pars(1.005)
L = n.arange(39.8,43.85, .01)
phi_z2 = n.zeros(n.size(L))
for i in range (0,n.size(L)):
    if L[i] <= n.log10(z2_pars[2]):
        phi_z2[i] = ((z2_pars[0]+1)*(L[i]-42.5)+z2_pars[3])
    if L[i] >= n.log10(z2_pars[2]):
        phi_z2[i] = ((z2_pars[1]+1)*(L[i]-42.5)+z2_pars[4])

p.plot(l,z2_plot,'o',color='g')
p.plot(L, phi_z2,color='g')
##
l = n.arange(41.6, 43.6, .15)
z3_plot = n.array([18.86, 22.3, 23.61, 16.6, 9.63, 4.52, 2.33, 1.15, .27, .34, .17, .06, .03, .03])
z3_plot = n.log10(z3_plot*10**-4)
z3_pars = rs.lum_pars(1.191)
L = n.arange(39.8,43.6, .01)
phi_z3 = n.zeros(n.size(L))
for i in range (0,n.size(L)):
    if L[i] <= n.log10(z3_pars[2]):
        phi_z3[i] = ((z3_pars[0]+1)*(L[i]-42.5)+z3_pars[3])
    if L[i] >= n.log10(z3_pars[2]):
        phi_z3[i] = ((z3_pars[1]+1)*(L[i]-42.5)+z3_pars[4])

p.plot(l,z3_plot,'o',color='b')
p.plot(L, phi_z3,color='b')
##
l = n.arange(41.75, 43.85, .15)
z4_plot = n.array([12.17, 15.53, 16.66, 9.70, 5.59, 3.99, 1.93, .87, .55, .11, .2, .13, .18, .03, .03])
z4_plot = n.log10(z4_plot*10**-4)
z4_pars = rs.lum_pars(1.349)
L = n.arange(39.8,43.85, .01)
phi_z4 = n.zeros(n.size(L))
for i in range (0,n.size(L)):
    if L[i] <= n.log10(z4_pars[2]):
        phi_z4[i] = ((z4_pars[0]+1)*(L[i]-42.5)+z4_pars[3])
    if L[i] >= n.log10(z4_pars[2]):
        phi_z4[i] = ((z4_pars[1]+1)*(L[i]-42.5)+z4_pars[4])

p.plot(l,z4_plot,'o',color='m')
p.plot(L, phi_z4,color='m')
