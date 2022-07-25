# -*- coding: utf-8 -*-
"""
Finite difference project
Author: Kee Onn Fong
"""

import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """ evaluates cos(x) + x sin (x) """
    return np.cos(x)+x*np.sin(x)

def func_dot(x):
    """ evaluates x cos (x) """
    return x*np.cos(x)

def fwddiff_of_func(x,stepsize):
    """
    Parameters
    ----------
    def fwddiff_of_func : evaluates the forward difference of func(x), which should be previously
    defined
    
    Inputs
    ------
    x: the x array of the original
    stepsize: the stepsize for evaulating (smaller -> smaller errors)

    Returns
    -------
    xplot: the x array for plotting 
    derivative: the y array for plotting the function
    """
    derivative = np.array([]) # initialize the derivative return array
    xplot = np.array([]) # initialize the plotting x array
    
    x_initial = np.min(x) # initial x is the lowest value of the x array
    x_fin = np.max(x)     # final x is the highest value of the x array
    xeval = x_initial     # evaulate starting from the initial 
    
    while xeval<=x_fin: 
        f = func(xeval) 
        fnext = func(xeval+stepsize)
        df = (fnext-f)/stepsize
        derivative = np.append(derivative,df)
        xplot = np.append(xplot, xeval) 
        xeval = xeval+stepsize
        
    return xplot, derivative # return the x array and the derivative for plotting

def bwdiff_of_func(x,stepsize):
    """
    Parameters
    ----------
    def bwdiff_of_func : evaluates the backward difference of func(x), which should be previously
    defined
    
    Inputs
    ------
    x: the x array of the original
    stepsize: the stepsize for evaulating (smaller -> smaller errors)

    Returns
    -------
    xplot: the x array for plotting 
    derivative: the y array for plotting the function
    """
    derivative = np.array([]) # initialize the derivative return array
    xplot = np.array([]) # initialize the plotting x array
    
    x_initial = np.min(x) # initial x is the lowest value of the x array
    x_fin = np.max(x)     # final x is the highest value of the x array
    xeval = x_initial     # evaulate starting from the initial 
    
    while xeval<=x_fin: 
        f = func(xeval) 
        fnext = func(xeval-stepsize) # changed here
        df = (f-fnext)/stepsize # changed here
        derivative = np.append(derivative,df)
        xplot = np.append(xplot, xeval) 
        xeval = xeval+stepsize
        
    return xplot, derivative # return the x array and the derivative for plotting

def cndiff_of_func(x,stepsize):
    """
    Parameters
    ----------
    def cndiff_of_func : Evaluates the backward difference of func(x), which should be previously
    defined. 
    
    Inputs
    ------
    x: the x array of the original
    stepsize: the stepsize for evaulating (smaller -> smaller errors)

    Returns
    -------
    xplot: the x array for plotting 
    derivative: the y array for plotting the function
    """
    derivative = np.array([]) # initialize the derivative return array
    xplot = np.array([]) # initialize the plotting x array
    
    x_initial = np.min(x) # initial x is the lowest value of the x array
    x_fin = np.max(x)     # final x is the highest value of the x array
    xeval = x_initial     # evaulate starting from the initial 
    
    while xeval<=x_fin: 
        f = func(xeval+stepsize) # changed here 
        fnext = func(xeval-stepsize) 
        df = (f-fnext)/(2*stepsize) # changed here
        derivative = np.append(derivative,df)
        xplot = np.append(xplot, xeval) 
        xeval = xeval+stepsize 
        
    return xplot, derivative # return the x array and the derivative for plotting


""" main script """
# define x-range
lowerlimit = -6
upperlimit = 6
n_points = 1000
stepsize = 0.25
x = np.linspace(lowerlimit,upperlimit,n_points)
f = func(x)
dfdx = func_dot(x)
xplot, yfwd = fwddiff_of_func(x,stepsize)
xplot, ybwd = bwdiff_of_func(x,stepsize)
xplot, ycntr = cndiff_of_func(x,stepsize)

""" plotting the function and its derivative"""
fig1 = plt.figure()
plt.plot(x,f,'.r')
plt.plot(x,dfdx,'-b')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend([r'$y = f(x)$',r'$y = f^\prime (x)$'])

""" plotting the function and the approximations"""
fig2 = plt.figure()
plt.plot(x,dfdx,':g')
plt.plot(xplot,yfwd,'>:r',markersize=5)
plt.plot(xplot,ybwd,'<:k',markersize=5)
plt.plot(xplot,ycntr,'x:b',markersize=5)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend([r'y exact',r'y forward difference',r'y backward difference',r'y central difference'])