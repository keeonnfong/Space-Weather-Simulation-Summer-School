using DifferentialEquations

# non-allocating reactor model
function reactor!(dx,x,p,t)
    k, S, rates = p
    rates[1] = k[1]*x[1]
    rates[2] = k[2]*x[2]
    rates[3] = k[3]*x[3]^2
    mul!(dx, S, rates)
end

# stoichiometry matrix
S = [-1 0 0;
     1 -1 1;
     0 2 -2]

# reaction coefficient
k = [100.0, 0.25, 1.0]

# initial condition
x0 = [1.0,0.0,0.0]

prob = ODEProblem(reactor!, x0, (0.0, 10.0), (k, S, zeros(3)))
@elapsed sol = solve(prob)