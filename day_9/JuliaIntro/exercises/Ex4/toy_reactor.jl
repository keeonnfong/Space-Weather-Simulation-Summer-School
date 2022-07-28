a = [[1/5],
     [3//40, 9//40], 
     [44//45, -56//15, 32//9], 
     [19372//6561, -25360//2187, 64448//6561, -212//729],  
     [9017//3168, -355//33, 46732//5247, 49//176, -5103//18656], 
     [35//384, 0, 500//1113, 125//192, -2187//6784, 11//84]]
b = [35//384, 0, 500//1113, 125//192, -2187//6784, 11//84, 0]
c = [0, 1//5, 3//10, 4//5, 8//9, 1, 1]

function explicit_RK_stepper(x,t,f,h,a,b,c)
    # complete the explicit_RK_stepper function (peak at the python code if you need help)
end

function integrate(f, x0, tspan, h, int)
    t, tf = tspan
    x = x0
    trajectory = [x0]
    ts = [t]
    while t < tf
        h_eff = min(h, tf-t)
        x = int(x,t,f,h_eff)
        t = min(t+h_eff, tf)
        push!(trajectory,x)
        push!(ts, t)
    end
    return trajectory, ts
end

function reaction_rates(x,k)
    return k .* x
end

function reactor(x,t,k,S)
    return S * reaction_rates(x,k)
end

S = [-1 0 0;
     1 -1 1;
     0 2 -2];

k = [100.0, 0.25, 1.0];

trajectory, ts = integrate((x,t) -> reactor(x,t,k,S), [1.0,0,0], (0.0,10.0), 1e-4, 
                           (x,t,f,h) -> explicit_RK_stepper(x,t,f,h,a,b,c));

t = @elapsed integrate((x,t) -> reactor(x,t,k,S), [1.0,0,0], (0.0,10.0), 1e-4, 
                                     (x,t,f,h) -> explicit_RK_stepper(x,t,f,h,a,b,c))

println("The reactor model simulated in $t seconds in Julia.")
