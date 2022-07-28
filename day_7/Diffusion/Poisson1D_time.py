#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
plt.close()

"Initial conditions"
u0 = 0
uf = 0
errorwithn = np.array([])
nlist = np.array([]) 
ntimes = 1 #number of times to double the number of N
    
"number of points"
N = 8 
dt = 1/4
time = np.arange(0,3+dt,dt)
nt = len(time)
Dx = 1/N
x = np.linspace(0,1,N+1)

" initialize the solutions"
U = np.zeros((N+1,nt+1))

for it in range(0,nt-1):
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))   
    F = 2*(2*x**2 + 5*x -2)*np.exp(x)
    
    "Temporal term"
    A = A + (1/dt)*np.diag(np.ones(N+1))
    F = F + U[:,it]/dt
    
    "impose boundary conditions at x = 0"
    A[0,:] = np.concatenate(([1],np.zeros(N))) # dirichlet BC
    A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2])) # neumann BC order 2
    
    "impose boundary conditions at x = end"
    F[0] = u0
    F[-1] = 0 
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    ua = 2*x*(3-2*x)*np.exp(x)
    U[:,it+1] = u

    "Plotting solution"
    fig1 = plt.figure()
    plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
    plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
    plt.legend(fontsize=12,loc='upper left')
    plt.grid()
    plt.xlabel("x",fontsize=16)
    plt.ylabel("u",fontsize=16)
    
    "Compute error"
    error = np.max(np.abs(u-ua)) 
    print("Linf error u: %g\n" % error)
    









