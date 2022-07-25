# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:02:38 2022

@author: keeonn
"""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits import mplot3d

def lorenz63(x, t, sigma, rho, beta):
    """ ODE right hand side. """
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1]-x[0])
    x_dot[1] = x[0]*(rho-x[2])-x[1]
    x_dot[2] = x[0]*x[1] - beta*x[2]
    return x_dot

""" end of defintions"""

"generate random integer values"
from numpy.random import seed
from numpy.random import randint
# seed random number generator
seed(1)
xinit = randint(-20,20,20)
yinit = randint(-30,30,20)
zinit = randint(0,50,20)
fig = plt.figure()
ax = plt.axes(projection='3d')

"problem set up"
sigma = 10
rho = 28
beta = 8/3
t0 = 0 #initial time
tf = 20 #final time
stepsize = 0.1
for i in range(20):   
    "initial condition"
    x0 = [xinit[i], yinit[i], zinit[i]]
        
    "evaluate exact solution"
    time = np.linspace(t0,tf,1000) #timespan
    y_true = odeint(lorenz63,x0,time,args=(sigma,rho,beta)) #solution
    
    "do 3d plots"
    xline = y_true[:,0]
    yline = y_true[:,1]
    zline = y_true[:,2]
    ax.plot3D(xline, yline, zline)