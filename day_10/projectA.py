#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_zz = f
Domain: [0,1]
BC: u(0) = 0, u'(1) = 0
with f = 2*(2*z^2 + 5*z - 2)*ezp(z)

Analytical solution: 2*z*(3-2*z)*ezp(z)

Finite differences (FD) discretization: second-order diffusion operator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Change boundary conditions (Neumann first and second-order)

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
from rad_tran_complete import *
import time

nAlts = 41

"Altitude points with a ghost cell at the top"
N = nAlts
z = np.linspace(100,500,N)
z = np.append(z,z[N-1]+np.average(np.diff(z)))
Dz = np.average(np.diff(z))

"Solution time-step"
dt = 300

sza_in_deg = 0  # solar zenith angle
kappa = 8e-8  # conductivity coefficient

"Initiate temperature with a ghost cell at the top"
temp = init_temp(np.linspace(100, 500+Dz, N+1))

error = 1
while error>1e-3:
    
    "Evaluate Q_euv and rho as a function of SZA, altitude, and temperature"
    Qeuv,rho = Qeuv_function_with_rho(sza_in_deg, z, temp) # diffusion eqn source term
    
    "System matrix and RHS term"
    A = (1/Dz**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    F = Qeuv/kappa
    
    "temporal term"
    A = A + (1/dt)*np.diag(np.ones(N+1))
    F = F + (1/dt)*temp # dT/dt 
    
    "Dirichlet Boundary condition at bottom"
    A[0,:] = np.concatenate(([1], np.zeros(N)))
    F[0] = 200
    
    "Neumann Boundary condition at top"
    A[N,:] = (1/Dz)*np.concatenate((np.zeros(N-2),[1/2, -2, 3/2]))
    F[N] = 0
    
    "Solution of the linear system AU=F"
    tempnew = np.linalg.solve(A,F)
    
    "Difference in solution"
    error=np.average(abs(tempnew-temp))
    temp = tempnew
    print('error : ', error)
    
"Plotting solution"
fig = plt.figure()
plt.plot(temp[1:N],z[1:N],':ob',linewidth=2,label='$\widehat{u}$')
#plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.label("z",fontsize=16)
plt.ylabel("Altitude (km)",fontsize=16)
plt.xlabel("Temp (K)",fontsize=16)
time.sleep(0.2)
#plt.xlim([199.7,200.1])
    









