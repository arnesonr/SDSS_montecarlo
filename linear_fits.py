import numpy as n

# Turnover luminosity
z = [.116, .837, 1.005, 1.191, 1.349]
Lto = [41.03, 41.68, 41.75, 41.93, 42.0]

x=n.zeros(5)

for i in range (0,5):
    x[i]= z[i]*Lto[i]
    a = n.sum(x)

for i in range (0,5):
    x[i]=z[i]**2
    c = n.sum(x)

m = (n.size(z)*a-n.sum(z)*n.sum(Lto))/(n.size(z)*c-n.sum(z)**2)
b = (n.sum(Lto) - m*n.sum(z))/n.size(z)

for i in range (0,5):
    x[i]=(Lto[i]-(m*z[i]+b))**2/n.size(z)
    chi2 = n.sum(x)

# Try log(z+1)
logz = [n.log10(1.116),n.log10(1.837),n.log10(2.005),n.log10(2.191),n.log10(2.349)]

for i in range (0,5):
    x[i]= logz[i]*Lto[i]
    a = n.sum(x)

for i in range (0,5):
    x[i]=logz[i]**2
    c = n.sum(x)

m_2 = (n.size(logz)*a-n.sum(logz)*n.sum(Lto))/(n.size(logz)*c-n.sum(logz)**2)
b_2 = (n.sum(Lto) - m_2*n.sum(logz))/n.size(logz)

for i in range (0,5):
    x[i]=(Lto[i]-(m_2*logz[i]+b_2))**2/n.size(logz)
    chi2_2 = n.sum(x)
#
# Alpha faint
#
alpha_f = [-1.67, -1.3]
z = [.116, 1.1]

x=n.zeros(2)

for i in range (0,2):
    x[i]= z[i]*alpha_f[i]
    a = n.sum(x)

for i in range (0,2):
    x[i]=z[i]**2
    c = n.sum(x)

m = (n.size(z)*a-n.sum(z)*n.sum(alpha_f))/(n.size(z)*c-n.sum(z)**2)
b = (n.sum(alpha_f) - m*n.sum(z))/n.size(z)

for i in range (0,2):
    x[i]=(alpha_f[i]-(m*z[i]+b))**2/n.size(z)
    chi2 = n.sum(x)

# Try log(z+1)
logz = [n.log10(1.116),n.log10(2.1)]

for i in range (0,2):
    x[i]= logz[i]*alpha_f[i]
    a = n.sum(x)

for i in range (0,2):
    x[i]=logz[i]**2
    c = n.sum(x)

m_2 = (n.size(logz)*a-n.sum(logz)*n.sum(alpha_f))/(n.size(logz)*c-n.sum(logz)**2)
b_2 = (n.sum(alpha_f) - m_2*n.sum(logz))/n.size(logz)

for i in range (0,2):
    x[i]=(alpha_f[i]-(m_2*logz[i]+b_2))**2/n.size(logz)
    chi2_2 = n.sum(x)
#
# Alpha bright
#
alpha_b = [-2.83, -2.95]
z = [.116, 1.1]

x=n.zeros(2)

for i in range (0,2):
    x[i]= z[i]*alpha_b[i]
    a = n.sum(x)

for i in range (0,2):
    x[i]=z[i]**2
    c = n.sum(x)

m = (n.size(z)*a-n.sum(z)*n.sum(alpha_b))/(n.size(z)*c-n.sum(z)**2)
b = (n.sum(alpha_b) - m*n.sum(z))/n.size(z)

for i in range (0,2):
    x[i]=(alpha_b[i]-(m*z[i]+b))**2/n.size(z)
    chi2 = n.sum(x)

# Try log(z+1)
logz = [n.log10(1.116),n.log10(2.1)]

for i in range (0,2):
    x[i]= logz[i]*alpha_b[i]
    a = n.sum(x)

for i in range (0,2):
    x[i]=logz[i]**2
    c = n.sum(x)

m_2 = (n.size(logz)*a-n.sum(logz)*n.sum(alpha_b))/(n.size(logz)*c-n.sum(logz)**2)
b_2 = (n.sum(alpha_b) - m_2*n.sum(logz))/n.size(logz)

for i in range (0,2):
    x[i]=(alpha_b[i]-(m_2*logz[i]+b_2))**2/n.size(logz)
    chi2_2 = n.sum(x)

# Beta bright
z1 = [1.116, 1.837, 2.005, 2.191, 2.349]
beta = [-5.2576, -4.01, -3.86, -3.67, -3.49]

log_z = n.log10(z1)
x = n.zeros(n.size(beta))
for i in range (0,5):
    x[i]=log_z[i]*beta[i]

a = n.sum(x)

m = (a - ((n.sum(beta)*n.sum(log_z))/n.size(beta)))/(n.sum(log_z**2)-(n.sum(log_z)**2/n.size(beta)))

b = (n.sum(beta) - m*n.sum(log_z))/n.size(beta)
for i in range (0,5):
    x[i]=(beta[i]-(m*log_z[i]+b))**2/n.size(log_z)
    chi2 = n.sum(x)
# Beta faint
z1 = [1.116,2.1]
beta = [-3.5524, -2.6916]

log_z = n.log10(z1)
x = n.zeros(n.size(beta))
for i in range (0,2):
    x[i]=log_z[i]*beta[i]

a=n.sum(x)

m=(a - ((n.sum(beta)*n.sum(log_z))/n.size(beta)))/(n.sum(log_z**2)-(n.sum(log_z)**2/n.size(beta)))

b = (n.sum(beta) - m*n.sum(log_z))/n.size(beta)
