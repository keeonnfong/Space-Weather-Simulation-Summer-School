"""
Solution of a 1D Poisson equation in spherical coordinates: -u_rr - (2/r)*u_r = f
    Domain: [r0,r0+1]   IMPORTANT r>0
    BC: u(r0) = u0, u'(r0+1) = 0
    with f = 2*(2*A(rt)/r + B(rt))*erp(rt), being A(rt) = 2*rt^2 + rt - 3, and B(rt) = 2*rt^2 + 5*rt -2
    and rt = r-r0
    
Analytical solution: 2*(rt)*(3-2*(rt))*exp(rt) + u0

Finite differences (FD) discretization: second-order daccuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Introduce first-order derivative with second-order accuracy

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
r0 = 13
Dr = 1/N
r = np.linspace(0,1,N+1) + r0
rN = np.concatenate((r, [r[N]+Dr]))
rNt = rN-r0

"Dirichlet boundary condition at r0"
u0 = 1

"Time parameters"
dt = 1/24
time = np.arange(0,3+dt,dt)
nt = np.size(time)

"Initialize solution U"
U = np.zeros((N+2,nt))
U[:,0] = rNt*(3-2*rNt)*np.exp(rNt)+u0

"Solution at each time step"
for it in range(0,nt-1):
    "System matrir and RHS term"
    A = (1/Dr**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
    "First-order term"
    A = A + (1/Dr)*(np.diag(1/rN[1:N+2],-1) - np.diag(1/rN[0:N+1],1))
    
    "New source term"
    Ar = 2*rNt**2 + rNt - 3
    Br = 2*rNt**2 + 5*rNt - 2
    F = 2*(2*Ar/rN + Br)*np.exp(rNt)*np.abs(np.cos(pi*time[it+1]))
    
    "Time derivative terms"
    A = A + (1/dt)*np.diag(np.ones(N+2))
    F = F + U[:,it]/dt
    
    "Boundary condition at r=0"
    A[0,:] = np.concatenate(([1], np.zeros(N+1)))
    F[0] = u0
    
    "Boundary condition at r=0"
    A[N+1,:] = (0.5/Dr)*np.concatenate((np.zeros(N-1),[-1, 0, 1]))
    F[N+1] = 0
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u

"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(r0, r0+1),ylim =(u0,u0+5)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    plt.plot(r,U[0:N+1,i])
    myAnimation.set_data(r, U[0:N+1,i])
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True)
