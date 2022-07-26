
"""
Created on Wed Jul 20 15:14:41 2022

@author: Simone Servadio
"""

"""This file teaches first order numerical differentiation using 
finite difference methods"""

"Auxiliarly functions"

import sys
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """generic function"""
    return np.cos(x)+x*np.sin(x)

def func_dot(x):
    """Derivative of the generic function"""
    return x*np.cos(x)



"Dispaly function and its derivative"
n_points = 1000     # number of points
x_in = -6           # start 
x_fin = -x_in       # symmetric domain
x = np.linspace(x_in,x_fin,n_points) # independent variable
y = func(x) #dependent variable
y_dot = func_dot(x) # derivative

fig1 = plt.figure()
plt.plot(x,y,'-r')
plt.plot(x,y_dot,'b-')
plt.grid()
plt.xlabel('x',fontsize = 16)
plt.legend([r'$y$',r'$\dot y$'],fontsize=16)
#sys.exit() #exit form the script










"FINITE DIFFERENCE"
step_size = 0.25  #define interval step for differentiation

fig2 = plt.figure() #plot the correct solution 
plt.plot(x,y_dot,'-k')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$\dot y$ truth'])

"Forward Finite Difference"
x0 = x_in                      # initialize first point
y_dot_forw = np.array([])      # initialize solution array 
x_forw = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    current_value = func(x0)                              #f_k
    following_value = func(x0+step_size)                  #f_k+1
    slope = (following_value-current_value)/step_size     #(f_k+1 - f_k)/h
    x0 = x0 + step_size           
    x_forw = np.append(x_forw, x0)
    y_dot_forw = np.append(y_dot_forw, slope)
    
    
plt.plot(x_forw[:-1],y_dot_forw,'-r')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward'])
#sys.exit()









"Backward Finite Difference"
x0 = x_in                      # initialize first point
y_dot_back = np.array([])      # initialize solution array 
x_back = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    current_value = func(x0)                              #f_k
    previous_value = func(x0-step_size)                   #f_k-1
    slope = (current_value-previous_value)/step_size      #(f_k - f_k-1)/h
    x0 = x0 + step_size
    x_back = np.append(x_back, x0)
    y_dot_back = np.append(y_dot_back, slope)
    
    
plt.plot(x_back[:-1],y_dot_back,'-b')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',r'$\dot y$ backward'])
#sys.exit()









"Central Finite Difference"
x0 = x_in                      # initialize first point
y_dot_cent= np.array([])       # initialize solution array 
x_cent = np.array([x_in])      # initialize step points

while x0 <= x_fin:
    following_value = func(x0+step_size)                  #f_k+1
    previous_value = func(x0-step_size)                   #f_k-1
    slope = (following_value-previous_value)/step_size/2  #(f_k+1 - f_k-1)/2h
    x0 = x0 + step_size
    x_cent = np.append(x_cent, x0)
    y_dot_cent = np.append(y_dot_cent, slope)
    
    
plt.plot(x_cent[:-1],y_dot_cent,'-g')
plt.legend([r'$\dot y$ truth',r'$\dot y$ forward',
            r'$\dot y$ backward',r'$\dot y$ central'])
sys.exit()














































