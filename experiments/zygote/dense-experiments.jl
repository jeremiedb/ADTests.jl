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