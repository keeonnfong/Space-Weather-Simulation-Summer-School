# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:03:32 2022

@author: keeonn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_free(x, t):
    """ ODE right hand side. Originally dy/dt = -2y """
    g = 9.81 #m/s2
    l = 3 # m, length
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = (-g/l)*np.sin(x[0])
    return x_dot

def pendulum_damped(x, t):
    """ ODE right hand side. Originally dy/dt = -2y """
    g = 9.81 #m/s2
    l = 3 # m, length
    damp = 0.3 #damping coefficient
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = (-g/l)*np.sin(x[0]) - damp*x[1]
    return x_dot

def RHS(x, t):
    xdot = pendulum_damped(x, t)
    return xdot

def RK4(t0,y0,tf,stepsize):
    """ 
    Given a function y(t) and an initial condition y(t_0) = y0
    RK4 does a fourth-order approximation for the differential equation y(t)
    y(t) has to be previously defined as 'RHS'
    
    Inputs
    ------
    t0: initial time
    y0: initial solution
    tf: final time for the solution 
    stepsize: the stepsize for evaulating (smaller -> smaller errors)

    Returns
    -------
    timeline: the time array for plotting
    solution: the array of y(t) for plotting
    """
    current_time = t0
    timeline = np.array([t0]) #the first timestep(0) is included
    solution = np.array([y0]) #the first solution is included
    current_value = y0 #use the first solution to help us find the next solution

    while current_time < tf-stepsize: 
        # solve ODE
        k1 = RHS(current_value, current_time)
        k2 = RHS(current_value+k1*(stepsize/2), current_time+(stepsize/2))
        k3 = RHS(current_value+k2*(stepsize/2), current_time+(stepsize/2))
        k4 = RHS(current_value+k3*stepsize, current_time+stepsize)
        next_value = current_value + (k1+2*k2+2*k3+k4)*stepsize/6
    
        # Save solution
        next_time = current_time+stepsize
        timeline = np.append(timeline, next_time)
        solution = np.append(solution, next_value)

        # initialize next step
        current_time = next_time
        current_value = next_value       
        
    return timeline, solution

"""End of definitions"""
"problem set up"

y0 = [np.pi/3,0] #initial condition
t0 = 0 #initial time
tf = 15 #final time
stepsize = 0.4

"evaluate exact solution"
time = np.linspace(t0,tf) #timespan
y_true = odeint(RHS,y0,time) #solution

"evaluate rk4 solution"
timeline,solution = RK4(t0,y0,tf,stepsize)
solution = np.reshape(solution,(np.size(timeline),2))

fig1 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y_true[:,0],'b-',linewidth = 2)
plt.plot(timeline,solution[:,0],'.r',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta(t)$')
plt.subplot(2,1,2)
plt.plot(time,y_true[:,1],'b-',linewidth = 2)
plt.plot(timeline,solution[:,1],'.r',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\dot{\theta}(t)$')

