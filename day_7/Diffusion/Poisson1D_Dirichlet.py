#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Change boundary conditions

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
%matplotlib qt
plt.close()

"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1,N+1)

"Dirichlet boundary condition at x=0, x=1"
u0 = 0

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = (3*x + x**2)*np.exp(x)

"Boundary condition at x=0"
A[0,:] = np.concatenate(([1], np.zeros(N)))
F[0] = u0

"Boundary condition at x=0"
A[N,:] = np.concatenate((np.zeros(N),[1]))
F[N] = u0

"Solution of the linear system AU=F"
u = np.linalg.solve(A,F)
ua = -x*(x-1)*np.exp(x)+u0

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.axis([0, 1,u0, u0+0.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)