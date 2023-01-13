using Revise
using Flux
using Tracker
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
Flux.@functor MyDense
# (m::MyDense)(x) = m.w * x .+ m.b
(m::MyDense)(x) = exp.(m.w * x .+ m.b)

my_loss(m, x) = sum(m(x))

m = MyDense(randn(h1, f) .* 0.01, randn(h1) .* 0.01)
x = randn(f, bs)

loss, grads = withgradient(model -> my_loss(model, x), m)
grads = gradient(model -> my_loss(model, x), m)

function tracker_test(m, x)
    loss, grads = withgradient(model -> my_loss(model, x), m)
    # return nothing
    return loss, grads
end

loss, grads = tracker_test(m, x)

#  66.045 ms (140 allocations: 107.02 MiB)
@btime tracker_test($m, $x);
