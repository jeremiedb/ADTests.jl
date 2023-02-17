using Revise
using ReverseDiff
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults

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

# rd_loss(m::Linear, x) = sum(m(x))
m = Linear(w1, b1)
rd_loss(w, b, x) = sum(exp.(w * x .+ b))

# some inputs and work buffers to play around with
inputs = (m.w, m.b, x)
results = (similar(m.w), similar(m.b))
all_results = map(DiffResults.GradientResult, results);
cfg = ReverseDiff.GradientConfig(inputs);

const f_tape = GradientTape(rd_loss, (m.w, m.b, x))
const compiled_f_tape = compile(f_tape)

# 47.326 ms (0 allocations: 0 bytes)
@btime gradient!($results, $compiled_f_tape, $inputs);
# 61.471 ms (113 allocations: 139.01 MiB)
@btime gradient!($results, $rd_loss, $inputs);
# 60.507 ms (119 allocations: 148.02 MiB)
@btime gradient($rd_loss, $inputs);

function rd_test(loss, m, x)
    val, g = grad(loss, m, x)
    return val, g
end

val, grads = yota_test(yota_loss, m, x);

# 43.431 ms (77 allocations: 89.01 MiB)
@btime val, grads = yota_test($yota_loss, $m, $x);
