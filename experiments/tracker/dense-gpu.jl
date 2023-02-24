using Revise
using Flux
using Tracker
using CUDA
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = CUDA.randn(h1, f) .* 0.01;
b1 = CUDA.randn(h1);
x = CUDA.randn(f, bs) .* 0.01;

struct MyDense{A,B}
    w::A
    b::B
end
Flux.@functor MyDense
# (m::MyDense)(x) = m.w * x .+ m.b
(m::MyDense)(x) = exp.(m.w * x .+ m.b)

my_loss(m, x) = sum(m(x))

m = MyDense(w1, b1)

loss, grads = withgradient(model -> my_loss(model, x), m)
grads = gradient(model -> my_loss(model, x), m)

function tracker_test(m, x)
    loss, grads = withgradient(model -> my_loss(model, x), m)
    # return nothing
    return loss, grads
end

loss, grads = tracker_test(m, x)

#  32.108 ms (581 allocations: 29.42 KiB)
@btime CUDA.@sync tracker_test($m, $x);
# 0.106737 seconds (585 CPU allocations: 29.812 KiB) (12 GPU allocations: 121.006 MiB, 0.12% memmgmt time)
CUDA.@time tracker_test(m, x);
