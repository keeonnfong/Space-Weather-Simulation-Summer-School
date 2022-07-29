Final Project
=============

## Project A

Solve and plot the temperature using steady-state heat equation using the 
data and functions from `day_7` and `day_8`:

$$
\frac{\partial^2 T_i}{\partial z^2} = - \frac{Q_{EUV,i}}{4 \cdot 10^{-4}}
$$

The boundary conditions should be $\partial T / \partial z = 0$ at the top and
$T=200$ K in the bottom.

## Project B

Using RK4 solve the below question for 24 hours:

$$
\frac{\partial T_i}{\partial t} = \frac{Q_{EUV,i}}{1500 \cdot \rho_i}
$$

Start with a 5 min time step. If you have time, make an animation and explain
what is seen.

## Project C

Solve the chemistry equation using an implicit method

$$
\frac{\partial N_i}{\partial t} + V\frac{\partial N_i}{\partial z} = S_i - L_i 
$$

Where $N_i$ is the number density of species i ($O^+$, $O_2^+$, $N_2^+$, $NO^2$), V=-10m/s, $S_i$ are the sources (one of which will be EUV), and $L_i$ are the losses. You will need to include electrons as a species, but you don't have to advect it or solve for the chemistry for it, since $e- = \sum N_i$.

You can get the ionization rates from the euv_37.csv file.  The chemical reactions and the reaction rates are specified in the Chemistry lecture towards the end.  Note that this includes $He^+$ and $H^+$ and others, which you can ignore.
