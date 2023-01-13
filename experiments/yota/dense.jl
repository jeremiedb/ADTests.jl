using Revise
using Yota
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

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

# 43.431 ms (77 allocations: 89.01 MiB)
@btime val, grads = yota_test($yota_loss, $m, $x);
