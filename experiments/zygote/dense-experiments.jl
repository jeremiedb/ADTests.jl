using Revise
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
using BenchmarkTools
import Flux
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

##############################################
# intermediate allocations test
##############################################
d0 = Flux.Dense(w1, b1, exp)

function d1(x, w, b)
    x1 = w * x
    x2 = x1 .+ b
    x3 = exp.(x2)
    return x3
end
function d2(x, w, b)
    return exp.((w * x) .+ b)
end
function d3(x, w, b)
    x1 = w * x
    x1 .= x1 .+ b
    x1 .= exp.(x1)
    return x1
end
function d4(x, w, b)
    x1 = w * x
    x1 .= exp.(x1 .+ b)
    return x1
end

@code_lowered d0(x)
@code_lowered d1(x, w1, b1)
@code_lowered d2(x, w1, b1)
@code_lowered d3(x, w1, b1)
@code_lowered d4(x, w1, b1)

# 18.940 ms (4 allocations: 32.00 MiB)
@btime d0($x);
# 20.806 ms (6 allocations: 48.00 MiB)
@btime d1($x, $w1, $b1);
# 18.661 ms (4 allocations: 32.00 MiB)
@btime d2($x, $w1, $b1);
# 18.191 ms (2 allocations: 16.00 MiB)
@btime d3($x, $w1, $b1);
# 17.029 ms (2 allocations: 16.00 MiB)
@btime d4($x, $w1, $b1);

##############################################
# forward test
##############################################
m = Flux.Dense(w1, b1, exp)
f0(x) = sum(m(x))

function f2(x, w, b)
    # return sum((w * x) .+ b)
    return sum(exp.((w * x) .+ b))
end

function f4(x, w, b)
    x1 = w * x
    x1 .= exp.(x1 .+ b)
    return sum(x1)
end

@code_lowered f0(x)
@code_lowered f2(x, w1, b1)
@code_lowered f4(x, w1, b1)

# 18.552 ms (5 allocations: 32.00 MiB)
@btime f0($x);
# 18.606 ms (4 allocations: 32.00 MiB)
@btime f2($x, $w1, $b1);
# 16.961 ms (2 allocations: 16.00 MiB)
@btime f4($x, $w1, $b1);


##############################
# grads tests
##############################
ps = Dict(:w => w1, :b => b1)
function g0(m, x)
    grads = gradient(model -> sum(model(x)), m)
    return grads[1]
end
function gA(f, x, ps)
    grads = gradient(p -> f(x, p[:w], p[:b]), ps)
    return grads[1]
end

grads0 = g0(m, x)
gradsA = gA(f2, x, ps)
gradsA[:w]
gradsA[:b]

# 46.822 ms (28 allocations: 105.00 MiB)
@btime g0($m, $x);
# 47.382 ms (74 allocations: 105.01 MiB)
@btime gA($f2, $x, $ps);

function g_custom_A(x, ps)
    # forward
    x1 = ps[:w] * x
    x2 = x1 .+ ps[:b]
    x3 = exp.(x2)
    l = sum(x1)
    # backward
    dl = ones(size(x3)) # resuses intermediate steps allocations
    dx3 = dl .* exp.(x2)
    dx2 = dx3
    db = reshape(sum(dx3, dims = 2), size(ps[:b]))
    dw = dx2 * x'
    return (l = l, dw = dw, db = db)
end
function g_custom_B(x, ps)
    # forward
    x1 = ps[:w] * x .+ ps[:b]
    x2 = exp.(x1)
    l = sum(x2)
    # backward
    x2 .= 1 # resuses intermediate forward steps allocations
    x1 .= x2 .* exp.(x1)
    dw = x1 * x'
    db = reshape(sum(x1, dims = 2), size(ps[:b]))
    return (l = l, dw = dw, db = db)
end
function g_custom_C(x, ps)
    # forward
    x1 = ps[:w] * x
    x1 .= exp.(x1 .+ ps[:b])
    l = sum(x1)
    # backward
    x1 .= 1 .* x1 # resuses intermediate forward steps allocations
    dw = x1 * x'
    db = reshape(sum(x1, dims = 2), size(ps[:b]))
    return (l = l, dw = dw, db = db)
end

grads_a = g_custom_B(x, ps)
grads_a[:dw]
grads_a[:db]
grads_a[:l]

grads_b = g_custom_B(x, ps)
grads_b[:dw]
grads_b[:db]
grads_b[:l]

grads_c = g_custom_C(x, ps)
grads_c[:dw]
grads_c[:db]
grads_c[:l]

# 42.532 ms (34 allocations: 81.00 MiB)
@btime grads_a = g_custom_A($x, $ps);
# 38.850 ms (29 allocations: 49.00 MiB)
@btime grads_b = g_custom_B($x, $ps);
# 26.699 ms (19 allocations: 17.00 MiB)
@btime grads_c = g_custom_C($x, $ps);
