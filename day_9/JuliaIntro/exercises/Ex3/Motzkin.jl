cd(@__DIR__)
include("dual_number_arithmetic.jl")

# The way we set things up, we can also evaluate gradients of vector-valued functions!
# As an example, we consider Motzkin's polynomial:
Motzkin(x,y) = x^4*y^4 + x^2*y^4 - 3*x^2*y^2 + 1
Motzkin_gradient(x,y) = [4*x^3*y^4 + 2*x*y^4 - 6*x*y^2;
                         4*x^4*y^3 + 4*x^2*y^3 - 6*x^2*y]

x = DualNumber(rand(), [1.0, 0])
y = DualNumber(rand(), [0, 1.0])

# check if the result is correct:
@show Motzkin(x,y).val == Motzkin(x.val, y.val)
@show Motzkin(x,y).grad == Motzkin_gradient(x.val,y.val) 