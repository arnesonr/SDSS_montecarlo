import numpy as n
import matplotlib.pyplot as p
import matplotlib.mlab as mlab
import mock_lens_funcs as mlf
from matplotlib import cm

llist=n.load("")
how_many = n.size(llist)/n.size(llist[0])

#mlist = mlf.modelable(llist)
m_num = n.size(mlist)/2

b=n.zeros(how_many)
b_d=list()
b_m=list()
q_l=n.zeros(how_many)
q_l_d=list()
q_l_m=list()
g=n.zeros(how_many)
g_d=list()
g_m=list()
Rs=n.zeros(how_many)
Rs_d=list()
Rs_m=list()
rho=n.zeros(how_many)
rho_d=list()
rho_m=list()
theta=n.zeros(how_many)
theta_d=list()
theta_m=list()
mu=n.zeros(how_many)
mu_d=list()
mu_m=list()
s=n.zeros(how_many)
s_d=list()
s_m=list()
detect=n.zeros(how_many)

for i in range (0,how_many):
    detect[i] = llist[i][8]

for i in range (0,how_many):
    b[i] = llist[i][0]
    q_l[i] = 1.-llist[i][1]
    g[i] = llist[i][2]
    theta[i] = llist[i][3]
    rho[i] = llist[i][4]
    Rs[i] = llist[i][5]
    mu[i] = llist[i][6]
    s[i] = llist[i][7]

for i in range (0,how_many):
    if detect[i]==1:
        b_d.append(llist[i][0]*.01)
        q_l_d.append(1.-llist[i][1])
        g_d.append(llist[i][2])
        Rs_d.append(llist[i][5]*.01)
        rho_d.append(llist[i][4])
        theta_d.append(llist[i][3])
        mu_d.append(llist[i][6])
        s_d.append(llist[i][7])

for i in range (0,m_num):
    if mlist[i][1]==1:
        b_m.append(llist[mlist[i][0]][0]*.01)
        q_l_m.append(1.-llist[mlist[i][0]][1])
        g_m.append(llist[mlist[i][0]][2])
        Rs_m.append(llist[mlist[i][0]][5]*.01)
        rho_m.append(llist[mlist[i][0]][4])
        theta_m.append(llist[mlist[i][0]][3])
        mu_m.append(llist[mlist[i][0]][6])
        s_m.append(llist[mlist[i][0]][7])


mu, sigma = 1, .5
num, bins, patches = p.hist(b,bins=50,align='mid',label='Sample',fill=0,range=(0,5),normed=1)
y = mlab.normpdf( bins, mu, sigma)
l = p.plot(bins, y, 'r--', linewidth=1,label='Parent')

p.plot((n.mean(b_m),n.mean(b_m)),(0,3),'r',lw=1,label='Modelable Mean')
p.plot((n.mean(b_d),n.mean(b_d)),(0,3),'b',lw=1,label='Detected Mean')
p.plot((n.mean(b),n.mean(b)),(0,3),color='black',lw=1,ls='dashed',label='Sample Mean')

b=b*.01
b=n.log10(b)
b_d=n.log10(b_d)
b_m=n.log10(b_m)

p.hist(b_m,bins=30,align='mid',label='Modelable',fill=0, ls='solid' ,normed=1,histtype='step',lw=1.4,color='black',range=(-1.5,1.5))
p.hist(b_d,bins=30,align='mid',label='Detected',fill=0, ls='dashed' ,normed=1,histtype='step',lw=1.4,color='black',range=(-1.5,1.5))
p.hist(b,bins=30,align='mid',label='Sample',fill=0, ls='dotted' ,normed=1,histtype='step',lw=1.4,color='gray',range=(-1.5,1.5))
p.legend()

p.xticks( n.arange(-2.0,2.5,.5), ('0.01','','0.01', '','1.0','','10.0','', '100.0') )
p.yticks( n.arange(0,3.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0', '','2.5', '', '3.0'))
p.xlabel('Einstein Radius (arcsec)')
p.ylabel('PDF')


m, sigma = 1, .2
num, bins, patches = p.hist(g,bins=50,align='mid',label='Parent',fill=0,range=(0,2),normed=1)
y = mlab.normpdf( bins, m, sigma)
l = p.plot(bins, y, 'r--', linewidth=1,label='Parent')

p.plot((n.mean(g_d),n.mean(g_d)),(0,2.5),'b',lw=1,label='Detected Mean')
p.plot((n.mean(g_m),n.mean(g_m)),(0,2.5),'r',lw=1,label='Modelable Mean')
p.plot((n.mean(g),n.mean(g)),(0,2.5),color='black',lw=1,ls='dashed',label='Sample Mean')

p.hist(g_d,bins=30,align='mid',label='Detected',range=(0,2),normed=1,fill=0,color='black',ls='dashed',histtype='step',lw=1.4)
p.hist(g_m,bins=30,align='mid',label='Modelable',range=(0,2),normed=1,fill=0,color='black',ls='solid',histtype='step',lw=1.4)
p.hist(g,bins=30,align='mid',label='Sample',fill=0,range=(0,2),normed=1,color='gray',histtype='step',ls='dotted',lw=1.4)
p.legend()

p.xticks( n.arange(0,2.2,.2), ('0','0.2','0.4', '0.6','0.8','1.0', '1.2','1.4', '1.6', '1.8','2.0') )
p.yticks( n.arange(0,3.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0', '','2.5','','3.0') )
p.xlabel('Lens Surface Density Power Law Index')
p.ylabel('PDF')


# plot the PDF of q (between 0.0 and 0.95) that follows f(E) = -(E^.4-.95^.4)*E^.4
x=n.arange(0,.95,.01)
y=-(x**.4-.95**.4)*x**.4
x=1-x
p.plot(x,y,'r--',linewidth=1,label='Parent')
p.legend()

p.hist(q_l_m,bins=20,label='Modelable',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='solid')
p.hist(q_l_d,bins=20,label='Detected',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='dashed')
p.hist(q_l,bins=20,align='mid',label='Sample',fill=0,range=(0,1),normed=1,color='gray',histtype='step',lw=1.4,ls='dotted')
p.legend(loc='upper left')
p.xticks( n.arange(0,1.1,.1), ('0','0.1','0.2', '0.3','0.4','0.5', '0.6','0.7', '0.8', '0.9','1.0') )
p.yticks( n.arange(0,2.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0') )
p.xlabel('Lens Axis Ratio')
p.ylabel('PDF')

Rs=Rs*.01
p.hist(Rs_m,bins=20,label='Modelable',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='solid')
p.hist(Rs_d,bins=20,label='Detected',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='dashed')
p.hist(Rs,bins=20,align='mid',label='Sample',fill=0,range=(0,1),normed=1,color='gray',histtype='step',lw=1.4,ls='dotted')
p.legend(loc='lower right')
p.yticks( n.arange(0,2.25,.25), ('0','0.25','0.5', '0.75','1.0','1.25', '1.5','1.75', '2.0') )
p.xticks( n.arange(0,1.1,.1), ('0','0.1','0.2', '0.3','0.4','0.5', '0.6','0.7', '0.8', '0.9','1.0') )
p.xlabel('Source Size (arcsec)')
p.ylabel('PDF')


p.hist(mu_d,bins=30,label='Detected',align='mid',range=(0,8),normed=1,fill=0,color='blue',histtype='step',lw=1.4)
p.hist(mu_m,bins=30,label='Modelable',align='mid',range=(0,8),normed=1,fill=0,color='red',histtype='step',lw=1.4)
p.hist(mu,bins=30,align='mid',label='Sample',fill=0,range=(0,8),normed=1,color='black',histtype='step',lw=1.4,ls='dotted')
p.legend()

p.hist(s_d,bins=20,label='Detected',align='mid',log=True,normed=1,fill=0,color='blue',histtype='step',lw=1.4)
p.hist(s_m,bins=20,label='Modelable',align='mid',log=True,normed=1,fill=0,color='blue',histtype='step',lw=1.4)
p.hist(s,bins=20,align='mid',label='Sample',log=True,fill=0,normed=1,color='black',histtype='step',lw=1.4,ls='dotted')
p.legend()


p.polar(theta,rho,'o',color='black',ms=2)
p.polar(theta_m,rho_m,'o',color='red',ms=4)
p.polar(theta_d,rho_d,'o',color='blue',ms=4)

p.plot(b,g,'co',ms=4,label='Sample')
p.plot(b_d,g_d,'ro',ms=4,label='Detected')
p.plot(b_m,g_m,'yo',ms=4,label='Modelable')
p.legend()


H, xedges, yedges = n.histogram2d(b, g, bins=(50,20))
extentH = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
p.imshow(H, extent=extentH, interpolation='nearest',cmap=cm.gray,origin='lower')
p.colorbar()
p.contourf(H,extent=extentH,cmap=cm.gray,origin='upper')

I, xedges, yedges = n.histogram2d(b_d, g_d, bins=(50,20))
extentI = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
p.imshow(I, extent=extentH, interpolation='nearest',cmap=cm.gray,origin='lower')
p.colorbar()
p.contourf(I,extent=extentI,cmap=cm.gray,origin='upper')

J, xedges, yedges = n.histogram2d(b_m, g_m, bins=(50,20))
extentJ = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
p.imshow(J, extent=extentH, interpolation='nearest',cmap=cm.gray,origin='lower')
p.colorbar()
p.contourf(J,extent=extentJ,cmap=cm.gray,origin='upper')

p.contourf(n.vstack((I,J)), extent=extentH, interpolation='nearest',cmap=cm.gray,origin='upper')

p.xticks(n.arange(0,2.5,.5),('0.0','0.5','1.0','1.5','2.0',))
p.yticks(n.arange(0,5.5,.5),('0.0','','1.0','','2.0','','3.0','','4.0','','5.0'))
p.ylabel('Einstein Radius (arcsec)')
p.xlabel('Lens Surface Density Power Law Index')
