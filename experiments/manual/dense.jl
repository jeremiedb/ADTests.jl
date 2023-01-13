using Revise
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

ps = Dict(:w => w1, :b => b1)

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