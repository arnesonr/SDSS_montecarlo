import distance_funcs as df
import numpy as n
a = [.115, .15, .09, .36, .16]
b = [.105, .08, .08, .14, .105]
z = [1.09,.69,.69,.85,.98]
f = [69.9, 8.7, 11.4, 44.9, 40.7]

ab = [.012075, .012, .0072, .0504, .0168]

L = n.ones(5)
for i in range (0,5):
    L[i]=4.0*n.pi*(df.Dl(z[i])*3.086E24)**2*f[i]

logL = n.log10(L)
logz = n.log10(z)
