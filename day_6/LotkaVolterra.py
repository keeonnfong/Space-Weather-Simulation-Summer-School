#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:30:53 2022

@author: home
"""

"""Data available for this script at 
https://github.com/stan-dev/example-models/tree/master/knitr/lotka-volterra
and useful presentation that explains the derivation of optimal values
https://jmahaffy.sdsu.edu/courses/f17/math636/beamer/lotvol-04.pdf"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint 

def Predator_vs_prey(x,t,prey_up,prey_down,pred_up,pred_down):
    xdot = np.zeros(2)
    xdot[0] = prey_up*x[0] - prey_down*x[0]*x[1] # prey equation
    xdot[1] = pred_down*x[0]*x[1]- pred_up*x[1]  # predator equation
    return xdot

"Load Population"
pop = pd.read_csv("hudson-bay-lynx-hare.csv")


"Dysplay population"
fig1 = plt.figure()
plt.plot(pop['Year'],pop['Hare'],'o-b')
plt.plot(pop['Year'],pop['Lynx'],'o-r')
plt.grid()
plt.xlabel('Years')
plt.ylabel('Population')

fig2 = plt.figure()
plt.plot(pop['Lynx'],pop['Hare'],'o-b')
plt.grid()
plt.xlabel('Lynx')
plt.ylabel('Hare')

"Propagation"
n_points = len(pop['Year'])
grow_hare = 0.48069
shrink_hare = 0.024822 
grow_lynx = 0.92718 
shrink_lynx = 0.027564

time = np.linspace(pop['Year'][0],pop['Year'].iat[-1],1000)

fig3 = plt.figure()
plt.grid()
for i in range(n_points):
    x0 = [pop['Hare'][i],pop['Lynx'][i]]
    y = odeint(Predator_vs_prey,x0,time,args=(grow_hare,shrink_hare,grow_lynx,shrink_lynx))
    plt.plot(y[:,1],y[:,0],'grey')

plt.plot(pop['Lynx'],pop['Hare'],'ob')



"Optimal Population"
x0_opt = [34.9134, 3.8566] # Optimal Initial Estimates
y = odeint(Predator_vs_prey,x0_opt,time,args=(grow_hare,shrink_hare,grow_lynx,shrink_lynx))
plt.plot(y[:,1],y[:,0],'r')

fig4 = plt.figure()
plt.plot(pop['Year'],pop['Hare'],'ob'); 
plt.plot(pop['Year'],pop['Lynx'],'or'); 
plt.plot(time,y[:,0],'b')
plt.plot(time,y[:,1],'r')
plt.grid()
plt.xlabel('Years')
plt.ylabel('Population')
plt.legend(['Hare truth','Lynx thruth','Hare model','Lynx model'])

"Average populations"
hare_ave_theory = grow_lynx/shrink_lynx
lynx_ave_theory = grow_hare/shrink_hare

hare_ave_data = pop["Hare"].mean()
lynx_ave_data = pop["Lynx"].mean()

hare_ave_prop = y[:,0].mean()
lynx_ave_prop = y[:,1].mean()

x0s = np.array([[hare_ave_theory, lynx_ave_theory],
                 [hare_ave_data, lynx_ave_data],
                 [hare_ave_prop,lynx_ave_prop]])

for i in range(3):
    y = odeint(Predator_vs_prey,x0s[i,:],time,args=(grow_hare,shrink_hare,grow_lynx,shrink_lynx))
    fig3.axes[0].plot(y[:,1],y[:,0],'b')
    fig4.axes[0].plot(time,y[:,0],'c')
    fig4.axes[0].plot(time,y[:,1],'m')
    
    
    
















