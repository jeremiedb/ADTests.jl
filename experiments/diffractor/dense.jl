using Revise
using ChainRulesCore
import ChainRulesCore: rrule
using Diffractor
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

struct MyDense{A,B}
    w::A
    b::B
end
# (m::MyDense)(x) = m.w * x .+ m.b
(m::MyDense)(x) = exp.(m.w * x .+ m.b)

my_loss(m, x) = sum(m(x))

m = MyDense(w1, b1)

loss, grads = withgradient(model -> my_loss(model, x), m)
grads = gradient(model -> my_loss(model, x), m)[1]

function diffractor_test(m, x)
    loss, grads = withgradient(model -> my_loss(model, x), m)
    # return nothing
    return loss, grads
end

loss, grads = diffractor_test(m, x)

#  
@btime diffractor_test($m, $x);
