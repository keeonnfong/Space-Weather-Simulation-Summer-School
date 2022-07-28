#!/usr/bin/env python
"""
Advection of a pollutant subject to a constant velocity

1D Convection-Diffusion equation: u_t -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u'(0) = 0, u(1) = u0
with f = 100*exp(-((x-0.8)/0.01)^2)*((sin(2*pi*t) + abs(sin(2*pi*t)))/2)

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme
    - First-order upwind
    - Limiters to switch from high to low-resolution
    
    
Tasks:
    - See what happens as we change time-step
    - See what happens as we change viscosity
    
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0
c = +2
u0 = 0

"Scheme parameters"
beta = 2
order = 2

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+2)
xN = np.concatenate(([x[0]-Dx],x))

"Time parameters"
dt = 1/80
time = np.arange(0,3+dt,dt)
nt = np.size(time)

"Initialize solution variable"
U = np.zeros((N+2,nt))

for it in range(nt-1):

    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))

    "Advection term:"
        
    "previous U"
    U0 = U[:,it]
       
    cp = max(c,0)
    cm = min(c,0)
    
    if order<2:
        Advp = cp*(np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1)) 
        Advm = cm*(np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),+1)) 
    else:
        Advp = cp*((3/2)*np.diag(np.ones(N+2)) - 2*np.diag(np.ones(N+1),-1) + (1/2)*np.diag(np.ones(N),-2)) 
        Advm = cm*((3/2)*np.diag(np.ones(N+2)) - 2*np.diag(np.ones(N+1),+1) + (1/2)*np.diag(np.ones(N),+2)) 
    Adv = (1/Dx)*(Advp-Advm)
    A = Diff + Adv
    
    "Source term"
    sine = np.sin(2*pi*time[it+1])
    sineplus = 0.5*(sine + np.abs(sine))
    F = 100*np.exp(-((x-0.2)/0.01)**2)*sineplus
    
    "Temporal terms"
    A = A + (1/dt)*np.diag(np.ones(N+2))
    F = F + U0/dt

    "Boundary condition at x=0"
    A[0,:] = (1/Dx)*np.concatenate(([1.5, -2, 0.5],np.zeros(N-1)))
    F[0] = 0

    "Boundary condition at x=1"
    #A[N,:] = np.concatenate((np.zeros(N),[1]))
    #F[N] = u0
    A[N+1,:] = (1/Dx**2)*np.concatenate((np.zeros(N-1),[1,-2,1])) # neumann BC order 2 with ghost
    F[N] = 0
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    #u[N] = u[N-2] # concentration just accumulates, doesn't reflect
    U[:,it+1] = u
    


"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(u0-1e-2,u0+0.5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = U[0:N+1,i]
    plt.plot(x[0:N+1],u)
    myAnimation.set_data(x[0:N+1], u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=False)
   
if nu>0:
    "Peclet number"
    P = np.abs(c*Dx/nu)
    print("Pe number Pe=%g\n" % P);

"CFL number"
CFL = np.abs(c*dt/Dx)
print("CFL number CFL=%g\n" % CFL);



