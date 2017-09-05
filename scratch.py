import numpy as n
from matplotlib import pyplot as p
import distance_funcs as df

elg1=n.load('ELG1.npy')
elg2=n.load('ELG2.npy')
elg3=n.load('ELG3.npy')
elg4=n.load('ELG4.npy')
elg5=n.load('ELG5.npy')
elg6=n.load('ELG6.npy')

num1=n.size(elg1)
num2=n.size(elg2)
num3=n.size(elg3)
num4=n.size(elg4)
num5=n.size(elg5)
num6=n.size(elg6)

z=list()
for i in range (0,num1/3):
    z.append(elg1[i][0])

for i in range (0,num2/3):
    z.append(elg2[i][0])

for i in range (0,num3/3):
    z.append(elg3[i][0])

for i in range (0,num4/3):
    z.append(elg4[i][0])

for i in range (0,num5/3):
    z.append(elg5[i][0])

for i in range (0,num6/3):
    z.append(elg6[i][0])

L=list()
for i in range (0,num1/3):
    L.append(elg1[i][1])

for i in range (0,num2/3):
    L.append(elg2[i][1])

for i in range (0,num3/3):
    L.append(elg3[i][1])

for i in range (0,num4/3):
    L.append(elg4[i][1])

for i in range (0,num5/3):
    L.append(elg5[i][1])

for i in range (0,num6/3):
    L.append(elg6[i][1])

beam_num=list()
for i in range (0,num1/3):
    beam_num.append(elg1[i][2])

for i in range (0,num2/3):
    beam_num.append(elg2[i][2])

for i in range (0,num3/3):
    beam_num.append(elg3[i][2])

for i in range (0,num4/3):
    beam_num.append(elg4[i][2])

for i in range (0,num5/3):
   beam_num.append(elg5[i][2])

for i in range (0,num6/3):
    beam_num.append(elg6[i][1])

x=p.hist(L,20)
dL=(x[1][1]-x[1][0])*n.log(10)
binsize=x[1][1]-x[1][0]
bins=n.size(x[0])
dn_dL_dv=list()
domega = (n.pi*(par[1]/2)**2) * (1.0/206265.0**2)
A = domega * par[3]
dVc = (((3000.0/0.7) * (1.+.2)**2)/ n.sqrt(0.3*(1.+.2)**3 + 0.7)) * df.Da(.2)**2 * par[0] * A
for i in range (0,bins-1):
    dn_dL_dv.append(n.log10(abs(x[0][i+1]-x[0][i])/(dL*dVc)))

logl=list()
for i in range (0,bins-1):
    logl.append(x[1][i])

p.bar(logl,dn_dL_dv,binsize,color='b')
z=n.arange(39.8,42.4,.2)
y=[-1.853,-1.8764,-2.0034,-2.0893,-2.2055,-2.3622,-2.5675,-2.856,-3.206,-3.614,-4.0809,-4.0969,-4.61979,-4.7959]
p.plot(z,y,'*',color='r')


a=list()
for i in range (0,46834):
    for j in range(0,46834):
        if beam_num[i]==beam_num[j] and i != j: a.append(j)

s=[20.0,19.,23.,19.,19.,25.,22.,62.,54.,12.,26.,29.,15.,31.,18.,18.,31.,52.,29.,9.,14.,16.,16.,23.,16.,16.,18.,10.,17.,26.,40.,30.,19.,27.,8.,29.,21.,23.,9.,10.,10.,14.,11.,12.,14.,19.,13.,53.,30.]
z = [.5952,.2708,.6322,.5248,.5011,.5235,.624,.5812,.3809,.5672,.324,.4538,.467,.47,.4797,.2764,.2512,.5809,.5466,.4646,.5435,.669,.4446,.4488,.33,.5141,.4425,.4814,.3556,.8115,.4404,.4857,.4317,.3956,.5388,.5241,.7933,.4357,.3776,.6672,.4635,.3964,.6306,.6029,.6238,.4635,.517,.2718,.7146]
r = [18.48,17.88,16.79,18.26,19.18,17.33,18.85,17.85,17.95,18.7,16.08,17.76,17.52,16.84,17.99,16.1,16.12,17.27,18.19,18.84,18.29,18.04,17.10,17.33,18.40,16.68,18.02,16.91,17.52,17.78,18.34,17.25,17.63,16.82,18.38,17.5,17.27,16.85,17.39,18.04,17.53,16.89,17.87,18.66,18.33,17.59,16.69,16.54,18.42]

L=n.ones(49)
for i in range (0,49):
    L[i]=(df.Dl(z[i])*3.086E24)**2*4.*n.pi*s[i]*10**-17

x=n.zeros(49)
for i in range (0,49):
    x[i]= r[i]*z[i]
    a = n.sum(x)

for i in range (0,49):
    x[i]=z[i]**2
    c = n.sum(x)

mz = (n.size(z)*a-n.sum(z)*n.sum(r))/(n.size(z)*c-n.sum(z)**2)
bz = (n.sum(r) - mz*n.sum(z))/n.size(z)

for i in range (0,49):
    x[i]=(r[i]-(mz*z[i]+bz))**2/n.size(z)
    chi2 = n.sum(x)

xz=n.arange(.2,.9,.02)
yz=mz*xz+bz
p.plot(xz,yz)

x=n.zeros(49)
for i in range (0,49):
    x[i]= r[i]*l[i]
    a = n.sum(x)

for i in range (0,49):
    x[i]=l[i]**2
    c = n.sum(x)

ml = (n.size(l)*a-n.sum(l)*n.sum(r))/(n.size(l)*c-n.sum(l)**2)
bl = (n.sum(r) - ml*n.sum(l))/n.size(l)

for i in range (0,49):
    x[i]=(r[i]-(ml*l[i]+bl))**2/n.size(l)
    chi2 = n.sum(x)

xl=n.arange(40.6,42.0,.1)
yl=ml*xl+bl
p.plot(xl,yl)
