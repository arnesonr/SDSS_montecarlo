import numpy as n
import matplotlib.pyplot as p
import matplotlib.mlab as mlab
from matplotlib import cm
import os

def analysis(sim_name,b_lim=0.71):
    """
    PURPOSE: Script to generate pertinent plots from mock lens simulation

    USAGE:  analysis(sim_name)
    
    ARGUMENTS:
         sim_name: string name of llist and mlist 2d arrays
         [b_lim]: the lower limit on the minimum Einstein radius in arcsec
         
    RETURNS:  saves figures of plots

    WRITTEN:  Ryan A. Arneson, U. of Utah, 2011
    """
    llist = n.load('llist_'+sim_name+'.npy')
    how_many = n.size(llist)/n.size(llist[0])
    
    tag = '_b'+str(b_lim)
    if b_lim==0.71:
        mlist = n.load('mlist_'+sim_name+'.npy')
    else:
        mlist = list()
        for i in range(0,how_many):
            if llist[i][8]==1: #it is detected
                if llist[i][0] > (b_lim*100.0) and llist[i][5] < (2.0*llist[i][0]):
                    mlist.append([i,1])#it is modelable
                else:
                    mlist.append([i,0])#it is NOT modelable
    
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
        b[i] = llist[i][0]
        q_l[i] = 1.-llist[i][1]
        g[i] = llist[i][2]
        theta[i] = llist[i][3]
        rho[i] = llist[i][4]
        Rs[i] = llist[i][5]
        mu[i] = llist[i][6]
        s[i] = llist[i][7]
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

    b=b*.01
    if sim_name[0]=='u':
        ###
        ###Using my twodhist function
        ###
        import histogram2d as h
        R = h.twodhist(b,g,25,10)
        S = h.twodhist(b_d,g_d,25,10)
        T = h.twodhist(b_m,g_m,25,10)
        extent = [0.0,5.0, 0.0, 2.0]
        p.subplot(311)
        p.subplots_adjust(hspace=.5)
        p.imshow(R, extent=extent, interpolation='nearest',cmap=cm.gray)
        p.title('Parent',fontsize=12)
        cbar=p.colorbar()
        cbar.set_label('PDF($\\theta_E$,$\\gamma$)',fontsize=10)
        p.xticks( n.arange(0,5.5,.5), ('0','','1.0', '','2.0','', '3.0','', '4.0', '','5.0') )
        p.yticks( n.arange(0,2.5,.5), ('1.0','','2.0', '','3.0') )
        p.ylabel('$\\gamma$',fontsize=10)
        p.xlabel('$\\theta_E$["]',fontsize=10)
        p.subplot(312)
        p.imshow(S, extent=extent, interpolation='nearest',cmap=cm.gray)
        p.title('Detected',fontsize=12)
        cbar=p.colorbar()
        cbar.set_label('PDF($\\theta_E$,$\\gamma$)',fontsize=10)
        p.xticks( n.arange(0,5.5,.5), ('0','','1.0', '','2.0','', '3.0','', '4.0', '','5.0') )
        p.yticks( n.arange(0,2.5,.5), ('1.0','','2.0', '','3.0') )
        p.ylabel('$\\gamma$',fontsize=10)
        p.xlabel('$\\theta_E$["]',fontsize=10)
        p.subplot(313)
        p.imshow(T, extent=extent, interpolation='nearest',cmap=cm.gray)
        p.title('Modelable',fontsize=12)
        cbar=p.colorbar()
        cbar.set_label('PDF($\\theta_E$,$\\gamma$)',fontsize=10)
        p.xticks( n.arange(0,5.5,.5), ('0','','1.0', '','2.0','', '3.0','', '4.0', '','5.0') )
        p.yticks( n.arange(0,2.5,.5), ('1.0','','2.0', '','3.0') )
        p.ylabel('$\\gamma$',fontsize=10)
        p.xlabel('$\\theta_E$["]',fontsize=10)
        p.savefig(sim_name+tag+'.png')
        p.close()
        
    else:
        b=n.log10(b)
        b_d=n.log10(b_d)
        b_m=n.log10(b_m)

        print '# detected =', n.size(b_d)
        print '# modelable =', n.size(b_m)
        print 'ave b[dex]=', n.mean(b)
        print 'ave b_d [dex]=', n.mean(b_d)
        print 'ave b_m [dex]=', n.mean(b_m)
        print 'b mean-bias [dex] =', (n.mean(b_m)-n.mean(b))
        print 'ave g =', n.mean(g)
        print 'ave g_d =', n.mean(g_d)
        print 'ave g_m =', n.mean(g_m)
        print 'g mean-bias =', (n.mean(g_m)-n.mean(g))

        p.hist(b_m,bins=30,align='mid',label='Modelable',fill=0, ls='solid' ,normed=1,histtype='step',lw=1.4,color='black',range=(-2.0,2.0))
        p.hist(b_d,bins=30,align='mid',label='Detected',fill=0, ls='dashed' ,normed=1,histtype='step',lw=1.4,color='black',range=(-2.0,2.0))
        p.hist(b,bins=30,align='mid',label='Parent',fill=0, ls='dotted' ,normed=1,histtype='step',lw=1.4,color='gray',range=(-2.0,2.0))
        p.legend()
        p.xticks( n.arange(-2.0,2.5,.5), ('0.01','','0.01', '','1.0','','10.0','', '100.0') )
        p.yticks( n.arange(0,3.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0', '','2.5', '', '3.0'))
        p.xlabel('Einstein Radius (arcsec)')
        p.ylabel('PDF')
        p.savefig('b_'+sim_name+tag+'.png')
        p.close()
        
        p.hist(g_d,bins=30,align='mid',label='Detected',range=(0,2),normed=1,fill=0,color='black',ls='dashed',histtype='step',lw=1.4)
        p.hist(g_m,bins=30,align='mid',label='Modelable',range=(0,2),normed=1,fill=0,color='black',ls='solid',histtype='step',lw=1.4)
        p.hist(g,bins=30,align='mid',label='Sample',fill=0,range=(0,2),normed=1,color='gray',histtype='step',ls='dotted',lw=1.4)
        p.legend()
        p.xticks( n.arange(0,2.2,.2), ('0','0.2','0.4', '0.6','0.8','1.0', '1.2','1.4', '1.6', '1.8','2.0') )
        p.yticks( n.arange(0,3.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0', '','2.5','','3.0') )
        p.xlabel('Lens Surface Density Power Law Index')
        p.ylabel('PDF')
        p.savefig('g_'+sim_name+tag+'.png')
        p.close()


        p.hist(q_l_m,bins=20,label='Modelable',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='solid')
        p.hist(q_l_d,bins=20,label='Detected',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='dashed')
        p.hist(q_l,bins=20,align='mid',label='Sample',fill=0,range=(0,1),normed=1,color='gray',histtype='step',lw=1.4,ls='dotted')
        p.legend(loc='upper left')
        p.xticks( n.arange(0,1.1,.1), ('0','0.1','0.2', '0.3','0.4','0.5', '0.6','0.7', '0.8', '0.9','1.0') )
        p.yticks( n.arange(0,2.25,.25), ('0','','0.5', '','1.0','', '1.5','', '2.0') )
        p.xlabel('Lens Axis Ratio')
        p.ylabel('PDF')
        p.savefig('q_'+sim_name+tag+'.png')
        p.close()

        Rs=Rs*.01
        p.hist(Rs_m,bins=20,label='Modelable',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='solid')
        p.hist(Rs_d,bins=20,label='Detected',align='mid',range=(0,1),normed=1,fill=0,color='black',histtype='step',lw=1.4,ls='dashed')
        p.hist(Rs,bins=20,align='mid',label='Sample',fill=0,range=(0,1),normed=1,color='gray',histtype='step',lw=1.4,ls='dotted')
        p.legend(loc='lower right')
        p.yticks( n.arange(0,2.25,.25), ('0','0.25','0.5', '0.75','1.0','1.25', '1.5','1.75', '2.0') )
        p.xticks( n.arange(0,1.1,.1), ('0','0.1','0.2', '0.3','0.4','0.5', '0.6','0.7', '0.8', '0.9','1.0') )
        p.xlabel('Source Size (arcsec)')
        p.ylabel('PDF')
        p.savefig('r_'+sim_name+tag+'.png')
        p.close()
    
    return 
