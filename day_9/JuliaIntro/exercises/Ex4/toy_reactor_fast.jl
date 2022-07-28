using UnPack, LinearAlgebra, BenchmarkTools

#########################
### setting up the computational infrastructure
#########################

# Dormand-Prince RK coefficients
a = [[1/5],
     [3/40, 9/40], 
     [44/45, -56/15, 32/9], 
     [19372/6561, -25360/2187, 64448/6561, -212/729],  
     [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656], 
     [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

# caching memory to avoid allocations
struct explicit_RK_cache{T,C}
    a::Vector{Vector{C}}
    b::Vector{C}
    c::Vector{C}
    ps::Tuple{Vector{T}, Matrix{Int64}, Vector{T}}
    n::Int
    ks::Vector{Vector{T}}
    x_intermediate::Vector{T}
    x_new::Vector{T}
    x_old::Vector{T}
end

# constructor for the cache
function explicit_RK_cache(x0,a,b,c,ps)
    return explicit_RK_cache(a, b, c, ps, length(a)+1,[similar(x0) for i in 1:length(a)+1], similar(x0), x0, copy(x0))
end

# non-allocating generic explicit RK stepper 
function explicit_RK_stepper!(cache, t, f!, h)
    @unpack a, b, c, ps, n, ks, x_intermediate, x_new, x_old = cache
    @. c += h
    f!(ks[1], x_new, t, ps)
    @. x_new += b[1] * ks[1]
    for i in 2:n
        x_intermediate .= x_old 
        for k in 1:i-1
            @. x_intermediate += a[i-1][k] * ks[k]
        end
        f!(ks[i], x_intermediate, c[i], ps)
        @. x_new += b[i] * ks[i]
    end
    x_old .= x_new
end

# integrate routine that works with the custom cache
function integrate(f!::F, cache, tspan, h) where F
    t, tf = tspan
    N = ceil(Int64,(tf-t)/h)
    ts = range(t, tf, length=N+1)
    h = step(ts)
    cache.a .*= h
    cache.b .*= h
    cache.c .*= h
    cache.c .+= t
    trajectory = [copy(cache.x_new)]
    for i in 2:N+1
        explicit_RK_stepper!(cache, ts[i], f!, h)
        push!(trajectory, copy(cache.x_new))
    end
    return trajectory, ts
end

# non-allocating reactor model
function reactor!(dx,x,t,ps)
    k, S, rates = ps
    rates[1] = k[1]*x[1]
    rates[2] = k[2]*x[2]
    rates[3] = k[3]*x[3]^2
    mul!(dx, S, rates)
end

#########################
### setting up simulation
#########################
# stoichiometry matrix
S = [-1 0 0;
     1 -1 1;
     0 2 -2];

# reaction coefficient
k = [100.0, 0.25, 1.0];

# initial condition
x0 = [1.0,0.0,0.0]

# simulation cache 
cache = explicit_RK_cache(x0,a,b,c,(k,S,zeros(3)))

# simulation horizon
tspan = (0.0,10.0)

# simulation step size
h = 1e-4

# compile and then time
integrate(reactor!, cache, tspan, h)
t = @elapsed integrate(reactor!, cache, tspan, h)
println("The reactor model simulated in $t seconds in Julia");