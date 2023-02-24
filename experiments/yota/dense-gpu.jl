using Revise
using Yota
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

struct Linear{A,B}
    w::A
    b::B
end

# (m::Linear)(x) = m.w * x .+ m.b
(m::Linear)(x) = exp.(m.w * x .+ m.b)

yota_loss(m::Linear, x) = sum(m(x))
m = Linear(w1, b1)
val, g = grad(yota_loss, m, x)

function yota_test(loss, m, x)
    val, g = grad(loss, m, x)
    return val, g
end

val, grads = yota_test(yota_loss, m, x);

# 30.486 ms (582 allocations: 30.45 KiB)
@btime CUDA.@sync val, grads = yota_test($yota_loss, $m, $x);
# 0.119126 seconds (592 CPU allocations: 30.781 KiB) (11 GPU allocations: 89.006 MiB, 0.12% memmgmt time)
CUDA.@time val, grads = yota_test(yota_loss, m, x);
