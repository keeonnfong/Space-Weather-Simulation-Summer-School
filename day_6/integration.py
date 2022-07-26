#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:31:23 2022

@author: Simone Servadio
"""

"""This file teaches numerical propagation using 
finite difference methods, in particular fixed step integration up to order 4"""


"Auxiliarly functions"
#%matplotlib qt
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

def RHS(x, t):
    """ODE Right hand side"""
    return -2*x


"Set the problem"
y0 = 3 # initial condition
t0 = 0 # initial time
tf = 2 # final time

"Evaluate exact solution"
time = np.linspace(t0,tf) # time spanned
y_true = odeint(RHS,y0,time) # solution

fig1 = plt.figure()
plt.plot(time,y_true,'k-',linewidth = 2)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend(['Truth'])
#sys.exit()

"Numerical integration"
step_size = 0.2 #value of the fixed step size

"First Order Runge-Kutta or Euler Method"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk1 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    slope = RHS(current_value, current_time)
    next_value = current_value + slope * step_size
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk1 = np.append(sol_rk1, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk1,'r-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1'])
#sys.exit()









"Second Order Runge-Kutta"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk2 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    next_value = current_value + k2 * step_size
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk2 = np.append(sol_rk2, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk2,'b-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2'])
#sys.exit()









"Fourth Order Runge-Kutta"
current_time = t0
timeline = np.array([t0])
current_value = y0
sol_rk4 = np.array([y0])

while current_time < tf-step_size:
    
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    k3 = RHS(current_value + k2*step_size/2, current_time + step_size/2)
    k4 = RHS(current_value + k3*step_size, current_time + step_size)
    next_value = current_value + (k1+2*k2+2*k3+k4) * step_size/6
    
    # Save Solution
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time)
    sol_rk4 = np.append(sol_rk4, next_value)
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
plt.plot(timeline,sol_rk4,'g-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2','Runge-Kutta 4'])
sys.exit()





























