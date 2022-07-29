using BenchmarkTools

function cos_approx(x, N)
    # approximation of cosine via power series expansion
    # inputs:
    #       - x : argument of cosine 
    #       - N : truncation order of the power series approximation
    # outputs:
    #       - cos_val : approximation of cos(x)
    #sum((-1)**n * x**(2*n)/(2*factorial(n)) for n in range(order+1))

    A = sum(((-1)^n)*(x^(2*n))/(factorial(2*n)) for n in 0:N)
    return A
end

cos_approx((π/3),(10))
cos(π/3)

@btime cos_approx($(π/3),$(10)) 
@btime cos($(π/3))
@btime cos(π/3)