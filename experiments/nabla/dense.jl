using Revise
using Nabla
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01; # stack overflow crash - ChainRules -- ???
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

struct Linear{A,B}
    w::A
    b::B
end

(m::Linear)(x) = exp.(m.w * x .+ m.b)

loss(m::Linear, x) = sum(m(x))
m = Linear(w1, b1)
val, g = grad(loss, m, x)

function yota_test(loss, m, x)
    val, g = grad(loss, m, x)
    return val, g
end

val, grads = yota_test(loss, m, x);

@btime val, grads = yota_test($loss, $m, $x);
