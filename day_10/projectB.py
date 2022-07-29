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
__author__ = 'Luci Baker, Kee Onn Fong, Yi Hui Tee'
__email__ = 'ljbak@uw.edu, kofong@uw.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
from rad_tran_complete import *
import time
plt.close()

"Parameters"
nAlts = 41
kappa = 8e-8 # thermal conductivity
cp = 1500 # coefficient of pressure

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
    
    "System matrix and RHS term"
    Qeuv,rho = Qeuv_function_with_rho(sza_in_deg[t], z, temp) # diffusion eqn source term
    rho_diag = rho #diagonal term for rho
    rho_subdiag = rho[1:N+1] #subdiagonal term for rho
    rho_supdiag = rho[0:N] #supdiagonal term for rho
    A_diag = np.diag(2*np.divide(1,rho_diag))
    A_subdiag = np.diag(-1*np.divide(1,rho_subdiag),-1)
    A_supdiag = np.diag(-1*np.divide(1,rho_supdiag),+1)
    A = (kappa/cp)*(1/Dz**2)*(A_diag + A_subdiag +A_supdiag) # compute the A matrix
    F = np.divide(Qeuv,rho)/cp  
        
    "temporal term"
    A = A + (1/dt)*np.diag(np.ones(N+1))
    F = F + (1/dt)*temp 
    
    "Dirichlet Boundary condition at bottom"
    A[0,:] = np.concatenate(([1], np.zeros(N)))
    F[0] = 200
    
    "Neumann Boundary condition at top"
    A[N,:] = (1/Dz)*np.concatenate((np.zeros(N-2),[1/2, -2, 3/2]))
    F[N] = 0
    
    "Solution of the linear system AU=F"
    tempnew = np.linalg.solve(A,F) # compute new temperature profile
    temparray[:,t+1] = tempnew # save temp to new timestep
    temp = temparray[:,t+1] # get temp from new timestep
    
"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim = (0,1000),ylim = (100,500)) 
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.xlabel("Temp",fontsize=16)
plt.ylabel("Altitude",fontsize=16)
plt.grid
def animate(i):
    
    u = temparray[0:N+1,i]
    plt.plot(u,z)
    print(f"Time: %d 00, SZA angle: %d" % (timeline[i]/3600.0,sza_in_deg[i]))
    myAnimation.set_data(u, z)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,np.size(timeline)),blit=True,repeat=False)




