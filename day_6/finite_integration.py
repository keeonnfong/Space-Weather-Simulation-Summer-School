# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:03:32 2022

@author: keeonn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(x, t):
    """ ODE right hand side. Originally dy/dt = -2y """
    return -2*x

def RK1(t0,y0,tf,stepsize):
    """ 
    Given a function y(t) and an initial condition y(t_0) = y0
    RK1 does linear approximation for the differential equation y(t)
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
        slope = RHS(current_value, current_time)
        next_value = current_value + slope * stepsize
    
        # Save solution
        next_time = current_time+stepsize
        timeline = np.append(timeline, next_time)
        solution = np.append(solution, next_value)

        # initialize next step
        current_time = next_time
        current_value = next_value       
        
    return timeline, solution

def RK2(t0,y0,tf,stepsize):
    """ 
    Given a function y(t) and an initial condition y(t_0) = y0
    RK2 does a second-order approximation for the differential equation y(t)
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
        k2 = RHS(current_value+(stepsize/2)*k1, current_time+(stepsize/2))
        next_value = current_value + k2 * stepsize
    
        # Save solution
        next_time = current_time+stepsize
        timeline = np.append(timeline, next_time)
        solution = np.append(solution, next_value)

        # initialize next step
        current_time = next_time
        current_value = next_value       
        
    return timeline, solution

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
y0 = 3 #initial condition
t0 = 0 #initial time
tf = 2 #final time
stepsize = 0.666666

"evaluate exact solution"
time = np.linspace(t0,tf) #timespan
y_true = odeint(RHS,y0,time) #solution

fig1 = plt.figure()
plt.plot(time,y_true,'k-',linewidth = 2)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(t)$')

"RK1 numerical integration"   
timeline,solution = RK1(t0,y0,tf,stepsize)
plt.plot(timeline,solution,'o-r',linewidth = 2)

"RK2 numerical integration"   
timeline,solution = RK2(t0,y0,tf,stepsize)
plt.plot(timeline,solution,'o-b',linewidth = 2)

"RK4 numerical integration"   
timeline,solution = RK4(t0,y0,tf,stepsize)
plt.plot(timeline,solution,'o-g',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2','Runge-Kutta 4'])