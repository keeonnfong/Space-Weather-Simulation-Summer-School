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
ntimes = 5 #number of times to double the number of N
for ghost_flag in [0,1]:
    
    N = 8 #number of points
    for i in range(1,ntimes+1,1): # repeat for any number of improvements in N points
        Dx = 1/N
        x = np.linspace(0,1,N+1)
        
        if ghost_flag:
            x = np.concatenate((x,[x[-1]+Dx]))
    
        "System matrix and RHS term"
        if ghost_flag:
            A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))   
        else: 
            A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))   
        
        F = 2*(2*x**2 + 5*x -2)*np.exp(x)
        
        "impose boundary conditions on A"
        
        if ghost_flag:
            A[0,:] = np.concatenate(([1],np.zeros(N+1))) # dirichlet BC
            A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1/2,0,1/2])) # neumann BC order 2 with ghost
        else:
            A[0,:] = np.concatenate(([1],np.zeros(N))) # dirichlet BC
            A[-1,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2])) # neumann BC order 2
        
        "impose boundary conditions on F"
        F[0] = u0
        F[-1] = 0 
        
        "Solution of the linear system AU=F"
        u = np.linalg.solve(A,F)
        ua = 2*x*(3-2*x)*np.exp(x)
        
        "Plotting solution"
        fig1 = plt.figure()
        plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
        plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
        plt.legend(fontsize=12,loc='upper left')
        plt.grid()
        plt.xlabel("x",fontsize=16)
        plt.ylabel("u",fontsize=16)
        
        "Compute error"
        if ghost_flag:
            error = np.max(np.abs(u[1:-1]-ua[1:-1])) # don't count the ghost point
        else:
            error = np.max(np.abs(u-ua)) 
        print("Linf error u: %g\n" % error)
        errorwithn = np.append(errorwithn,error)
        nlist = np.append(nlist,N)
        N = N*2
        
"compute error difference"
nlist = np.reshape(nlist,(2,ntimes))
errorwithn = np.reshape(errorwithn,(2,ntimes))
fig1 = plt.figure()
plt.plot(nlist[0,1:],errorwithn[0,1:]/errorwithn[0,0:-1],'x-r')
plt.plot(nlist[1,1:],errorwithn[1,1:]/errorwithn[1,0:-1],'x-b')
plt.xlabel('Number of points, N')
plt.ylabel('Error(N)/Error(N-1)')
plt.legend(['With ghost point (2nd order)','Without ghost point (2nd order)'])








