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
%matplotlib inline
plt.close()

"Number of points"
N = 8
u0 = 0
uf = 0

for order in [1,2]: # try 1st and 2nd order for neumann BC
    errorwithn = np.array([])
    nlist = np.array([])    
    for i in [1,2,3,4,5]: # repeat for any number of improvements in N points
        Dx = 1/N
        x = np.linspace(0,1,N+1)
        
        "System matrix and RHS term"
        A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))   
        #F = (3*x + x**2)*np.exp(x) for old code
        F = 2*(2*x**2 + 5*x -2)*np.exp(x)
        
        "impose boundary conditions at x = 0"
        A[0,:] = np.concatenate(([1],np.zeros(N))) # dirichlet BC
        #A[-1,:] = np.concatenate((np.zeros(N),[1])) # dirichlet BC
        if order==1:
            A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1])) # neumann BC order 1
        elif order==2:
            A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2])) # neumann BC order 2
        
        "impose boundary conditions at x = end"
        F[0] = u0
        #F[-1] = uf for old code
        F[-1] = 0 
        
        "Solution of the linear system AU=F"
        U = np.linalg.solve(A,F)
        u = U
        #ua = -x*(x-1)*np.exp(x)+uf for old code
        ua = 2*x*(3-2*x)*np.exp(x)
        
        "Plotting solution"
        fig1 = plt.figure()
        plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
        plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
        plt.legend(fontsize=12,loc='upper left')
        plt.grid()
        #plt.axis([0, 1,0, 0.5])
        plt.xlabel("x",fontsize=16)
        plt.ylabel("u",fontsize=16)
        
        "Compute error"
        error = np.max(np.abs(u-ua))
        print("Linf error u: %g\n" % error)
        errorwithn = np.append(errorwithn,error)
        nlist = np.append(nlist,N)
        N = N*2
        
    "compute error difference"
    #fig1 = plt.figure()
    #plt.plot(nlist[1:],errorwithn[1:]/errorwithn[0:-1],'x-r')
    #plt.xlabel('Number of points, N')
    #plt.ylabel('Error(N)/Error(N-1)')
    
    
    







