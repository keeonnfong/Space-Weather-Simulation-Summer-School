from numpy.linalg import norm
import numpy as np

def explicit_RK_stepper(f,x,t,h,a,b,c):
    """
        Implementation of generic explicit Runge-Kutta update for explicit ODEs
        
        inputs:
            x - current state 
            t - current time
            f - right-hand-side of the (explicit) ODE to be integrated (signature f(x,t))
            h - step size 
            a - coefficients of Runge-Kutta method (organized as list-of-list (or vector-of-vector))
            b - weights of Runge-Kutta method (list/vector)
            c - nodes of Runge-Kutta method (including 0 as first node) (list/vector)

        outputs: 
            x_hat - estimate of the state at time t+h
    """
    k = np.zeros(len(c)) # k vector with length of c (RK-c)
    k[0] = f(x,t) #1st k is just evaluating the function at x and t
    for i in range(1, len(c)):
        delta_x = h*sum(a[i-1]*k[0:i]) # h times sum of a vectors and k bectors 
        delta_t = c[i]*h #the timestep 
        k[i] = f(x + delta_x, t + delta_t) # generate the new k vector
        
    x_hat = x + h*sum(b*k) # RK-len(c) prediction
    return x_hat
    

def integrate(f, x0, tspan, h, step):
    """
        Generic integrator interface

        inputs:
            f     - rhs of ODE to be integrated (signature: dx/dt = f(x,t))
            x0    - initial condition (numpy array)
            tspan - integration horizon (t0, tf) (tuple)
            h     - step size
            step   - integrator with signature: 
                        step(f,x,t,h) returns state at time t+h 
                        - f rhs of ODE to be integrated
                        - x current state
                        - t current time 
                        - h stepsize

        outputs: 
            ts - time points visited during integration (list)
            xs - trajectory of the system (list of numpy arrays)
    """
    t, tf = tspan
    x = x0
    trajectory = [x0]
    ts = [t]
    while t < tf:
        h_eff = min(h, tf-t)
        x = step(f,x,t,h_eff)
        t = min(t+h_eff, tf)
        trajectory.append(x)
        ts.append(t)
    return trajectory, ts