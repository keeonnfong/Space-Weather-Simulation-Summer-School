#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:17:19 2022

@author: Simone Servadio
"""

"Auxiliarly Functions"

#%matplotlib qt
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

def pendulum_free(x,t):
    """Dynamics of the pendulum without any constraint"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0])
    return xdot

def pendulum_damped(x,t):
    """Dynamics of the pendulum with a damper"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    damp = 0.3 #damper coefficient
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) - damp*x[1]
    return xdot

def pendulum_controlled(x,t,u):
    """Dynamics of the pendulum without an actuator that gives control torque"""
    g = 9.81 # gravity constant
    l = 3 #length of pendulum
    m = 0.2 # mass of ball
    xdot = np.zeros(2)
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) + u/m/l/l
    return xdot

def RK1(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 1"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #time step
        k1 = func(y[i], t[i]) #slope
        y[i+1] = y[i] + k1 * h #forward integration
    return y

def RK2(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 2"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #times tep
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2)
        y[i+1] = y[i] + k2 * h
    return y

def RK4(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 2"""
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i] #time - step
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1 * h / 2., t[i] + h / 2)
        k3 = func(y[i] + k2 * h / 2., t[i] + h / 2)
        k4 = func(y[i] + k3 * h, t[i] + h)
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y
    



"Propagate Free Pendulum"
x0 = np.array([np.pi/3, 0])
t0 = 0.0
tf = 15.0
n_points = 1000
time = np.linspace(t0,tf,n_points)
y = odeint(pendulum_free,x0,time)
y_rk = RK4(pendulum_free,x0,time)

fig1 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'b-',linewidth = 2)
plt.plot(time,y_rk[:,0],'c-.',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.legend(['odeint','rk4'])
plt.subplot(2,1,2)
plt.plot(time,y[:,1],'r-',linewidth = 2)
plt.plot(time,y_rk[:,1],'m-.',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
plt.legend(['odeint','rk4'])
#sys.exit()









"Propagate Damped Pendulum"
x0 = np.array([np.pi/3, 0])
n_points = 25
time_new = np.linspace(t0,tf,n_points)
y = odeint(pendulum_damped,x0,time)
#y_rk1 = RK1(pendulum_damped,x0,time_new)
y_rk2 = RK2(pendulum_damped,x0,time_new)
y_rk4 = RK4(pendulum_damped,x0,time_new)

"Display"
fig2 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'k-',linewidth = 2)
#plt.plot(time_new,y_rk1[:,0],'r-',linewidth = 2)
plt.plot(time_new,y_rk2[:,0],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,0],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(time,y[:,1],'k-',linewidth = 2)
#plt.plot(time_new,y_rk1[:,1],'r-',linewidth = 2)
plt.plot(time_new,y_rk2[:,1],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,1],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
sys.exit()









"Add A PD controller "
Kp = 2  # PD Proportional gain
Kd = 2  # PD derivative gain

n_points = 100
delta_read = 0.01 # frequency of observations

"Control Pendulum To Stable Equilibrium [0,0]"
timeline = np.array([t0])  #initial value is initial condition
x_hyst = np.array([x0])  #initial value is initial condition

t_in = t0  #initialize time
x_in = x0  #initialize state

u = -Kp*x0[0] - Kd*x0[1]; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    u = -Kp*x_fin[0] - Kd*x_fin[1];
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin]) #stack current state to solution vector
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig3 = plt.figure()
plt.subplot(2,1,1)
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
#sys.exit()









"Control Pendulum To Untable Equilibrium [pi,0]"
Kp = 10  # PD Proportional gain
Kd = 5  # PD derivative gain

timeline = np.array([t0])
x_hyst = np.array([x0]);

t_in = t0  #initialize
x_in = x0

x_des = np.array([np.pi,0])

u = -Kp*(x0[0]-x_des[0]) - Kd*(x0[1]--x_des[1]) ; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    u = -Kp*(x_fin[0]-x_des[0])  - Kd*(x_fin[1]-x_des[1]) ;
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin])
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig4 = plt.figure()
plt.subplot(2,1,1)
plt.axhline(y=np.pi,linewidth = 0.5,color = 'r')
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
#sys.exit()









"Control Pendulum To Untable Equilibrium [pi,0] and make it do a loop at t=7.5 seconds"
timeline = np.array([t0])
x_hyst = np.array([x0]);

t_in = t0  #initialize
x_in = x0

x_des_1 = np.array([np.pi,0])  #initial desired equilibrium state
x_des_2 = np.array([-np.pi,0]) # final desired equilibrium state
t_switch = 7.5 #time at which the pendulum rotates

u = -Kp*(x0[0]-x_des_1[0]) - Kd*(x0[1]--x_des_1[1]) ; #initial control

while t_in < tf:
    
    # Advance in time
    t_fin = t_in + delta_read
    
    # Solve ODE
    time = np.linspace(t_in,t_fin,n_points)
    y = odeint(pendulum_controlled,x_in,time,args=(u,))
    x_fin = y[-1,:]
    
    # Evaluate Control
    if t_fin <= t_switch:
        u = -Kp*(x_fin[0]-x_des_1[0])  - Kd*(x_fin[1]-x_des_1[1]) 
    else:
        u = -Kp*(x_fin[0]-x_des_2[0])  - Kd*(x_fin[1]-x_des_2[1]) 
    
    # Save Solution
    timeline = np.append(timeline, t_fin)
    x_hyst = np.vstack([x_hyst,x_fin])
    
    # Initialize Next Timestep
    x_in = x_fin;
    t_in = t_fin;

# Display
fig5 = plt.figure()
plt.subplot(2,1,1)
plt.axhline(y=np.pi,linewidth = 0.5,color = 'r')
plt.axhline(y=-np.pi,linewidth = 0.5,color = 'r')
plt.plot(timeline,x_hyst[:,0],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(2,1,2)
plt.plot(timeline,x_hyst[:,1],'b-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot \theta$')
sys.exit()














