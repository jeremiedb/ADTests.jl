using Revise
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
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
y = CUDA.randn(h1, bs) .* 0.01;

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
grads[:w]

function zygote_test(m, x)
    loss, grads = withgradient(model -> my_loss(model, x), m)
    # return nothing
    return loss, grads
end

loss, grads = zygote_test(m, x)

#  32.538 ms (585 allocations: 29.70 KiB)
@btime CUDA.@sync zygote_test($m, $x);
# 0.088988 seconds (587 CPU allocations: 29.766 KiB) (12 GPU allocations: 121.006 MiB, 0.13% memmgmt time)
CUDA.@time zygote_test(m, x);

# my_loss2(m, x, y) = sum(m(x) .* y)
# function zygote_test2(m, x, y)
#     loss, grads = withgradient(model -> my_loss2(model, x, y), m)
#     # return nothing
#     return loss, grads
# end
# loss, grads = zygote_test2(m, x, y)

# #  47.057 ms (28 allocations: 105.00 MiB)
# @btime zygote_test2($m, $x, $y);
