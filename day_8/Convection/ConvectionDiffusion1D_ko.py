#!/usr/bin/env python
"""
Solution of a 1D Convection-Diffusion equation: -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = 1

Analytical solution: (1/c)*(x-((1-exp(c*x/nu))/(1-exp(c/nu))))

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = 2
Narray = [32]
error = np.zeros(len(Narray))
P = error
order = 1

for ct in range(np.size(Narray)):
    "Number of points"
    N = Narray[ct]
    Dx = 1/N
    x = np.linspace(0,1,N+1)
    
    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))
    
    if order<2:
        "Advection term: first order upwind"
        cp = max(c,0)
        cm = min(c,0)
        Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1)) 
        Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),+1)) 
        
    else:
        "Advection term: centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1)
        Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    Adv = (1/Dx)*(Advp-Advm)
    A = Diff + Adv
    
    "Source term"
    F = np.ones(N-1)        
    
    "Solution of the linear system AU=F"
    U = np.linalg.solve(A,F)
    u = np.concatenate(([0],U,[0]))
    ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))
    
    fig = plt.figure()
    plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
    plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
    plt.legend(fontsize=12,loc='upper left')
    plt.grid()
    #plt.axis([0, 1,0, 2/c])
    plt.xlabel("x",fontsize=16)
    plt.ylabel("u",fontsize=16)
    
    
    "Compute error"
    error[ct] = np.max(np.abs(u-ua))
    print("Linf error u: %g\n" % error[ct])

    "Peclet number"
    P[ct] = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P[ct]);

fig = plt.figure()
error2over1 = error[1:len(error)]/error[0:len(error)-1]
plt.plot(Narray[1:len(Narray)],error2over1,'.-r',label = '$error_N\,/\,error_{N-1}$')
plt.plot(Narray,P,'.-b',label = 'Peclet number')
plt.legend(fontsize=12)