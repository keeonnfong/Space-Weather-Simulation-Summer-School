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
plt.close()

nAlts = 41

"Number of points"
N = nAlts
z = np.linspace(100,500,N)
z = np.append(z,z[N-1]+np.average(np.diff(z)))
Dz = np.average(np.diff(z))

"Initiate temperature"
temp = init_temp(np.linspace(100, 500+Dz, N+1))

"System matrix and RHS term"
A = (1/Dz**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = Qeuv_function(0, z, temp)/(8e-8) # source term

"Dirichlet Boundary condition at bottom"
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = 200

"Neumann Boundary condition at top"
A[N,:] = (1/Dz)*np.concatenate((np.zeros(N-2),[1/2, -2, 3/2]))
F[N] = 0

"Solution of the linear system AU=F"
temp = np.linalg.solve(A,F)

"Plotting solution"
plt.plot(temp,z,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.label("z",fontsize=16)
plt.ylabel("Altitude (km)",fontsize=16)
plt.xlabel("Temp (K)",fontsize=16)
#plt.xlim([199.7,200.1])

