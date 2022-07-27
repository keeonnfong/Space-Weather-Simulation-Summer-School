#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = 0, u'(1) = 0
with f = 2*(2*x^2 + 5*x - 2)*exp(x)

Analytical solution: 2*x*(3-2*x)*exp(x)

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
%matplotlib qt
plt.close()

"Number of points"
N = 16
Dx = 1/N
x = np.linspace(0,1,N+1)

"Dirichlet boundary condition at x=0"
u0 = 0

"Order of Neumann boundary condition approximation"
order = 2

if order<2:
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    F = 2*(2*x**2 + 5*x - 2)*np.exp(x)
    
    "Boundary condition at x=0"
    A[0,:] = np.concatenate(([1], np.zeros(N)))
    F[0] = u0
    
    "Boundary condition at x=0"
    A[N,:] = np.concatenate((np.zeros(N-1),[-1, 1]))
    F[N] = 0
else:
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
    xN = np.concatenate((x, [x[N]+Dx]))
    F = 2*(2*xN**2 + 5*xN - 2)*np.exp(xN)
    
    "Boundary condition at x=0"
    A[0,:] = np.concatenate(([1], np.zeros(N+1)))
    F[0] = u0
    
    "Boundary condition at x=0"
    A[N+1,:] = (0.5/Dx)*np.concatenate((np.zeros(N-1),[-1, 0, 1]))
    F[N+1] = 0

"Solution of the linear system AU=F"
u = np.linalg.solve(A,F)
u = u[0:N+1]
ua = 2*x*(3-2*x)*np.exp(x)+u0

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)