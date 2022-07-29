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
plt.close()

nAlts = 41

"Number of altitude points"
N = nAlts
z = np.linspace(100,500,N)
z = np.append(z,z[N-1]+np.average(np.diff(z)))
Dz = np.average(np.diff(z))

"Number of time points"
t0 = 0
tf = 86400 # seconds
dt = 5*60
timeline = np.arange(t0, tf, dt)

sza_in_deg = np.linspace(-180, 180, len(timeline))

"Initiate temperature"
temparray = np.zeros([N+1,len(timeline)])
temparray[:,0] = init_temp(np.linspace(100, 500+Dz, N+1))
temp = temparray[:,0]

for t in range(len(timeline)-1):
    
    error = 1
    while error>0.01:
        
        "System matrix and RHS term"
        A = (1/Dz**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
        Qeuv,rho = Qeuv_function_with_rho(sza_in_deg[t], z, temp) # diffusion eqn source term
        F = np.divide(Qeuv,rho*1500) # Qeuv/(8e-8) ?
        
        "temporal term"
        A = A*np.divide(8e-8,1500*rho) + (1/dt)*np.diag(np.ones(N+1))
        F = F + (1/dt)*temp # dT/dt + temporal source term
        
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
        #print('error : ', error)
    
    print('time (s) : ', timeline[t])
    "System matrix and RHS term"
    A = (1/Dz**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    Qeuv,rho = Qeuv_function_with_rho(sza_in_deg[t], z, temp) # diffusion eqn source term
    F = np.divide(Qeuv,rho*1500)
    
    "temporal term"
    A = A*np.divide(8e-8,1500*rho) + (1/dt)*np.diag(np.ones(N+1))
    F = F + (1/dt)*temp # dT/dt + temporal source term
    
    "Dirichlet Boundary condition at bottom"
    A[0,:] = np.concatenate(([1], np.zeros(N)))
    F[0] = 200
    
    "Neumann Boundary condition at top"
    A[N,:] = (1/Dz)*np.concatenate((np.zeros(N-2),[1/2, -2, 3/2]))
    F[N] = 0
    
    "Solution of the linear system AU=F"
    tempnew = np.linalg.solve(A,F)
    temparray[:,t+1] = tempnew
    temp = temparray[:,t+1]
    
"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim = (0,2000),ylim = (100,500)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.xlabel("Temp",fontsize=16)
plt.ylabel("Altitude",fontsize=16)
def animate(i):
    
    u = temparray[0:N+1,i]
    plt.plot(u,z)
    ax.set_title(f"SZA angle: %d" % sza_in_deg[i])
    print(f"SZA angle: %d" % sza_in_deg[i])
    myAnimation.set_data(u, z)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,np.size(timeline)),blit=True,repeat=False)




