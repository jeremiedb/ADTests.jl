using Revise
using BenchmarkTools
using Statistics
using Random: seed!
using LoopVectorization
using Tullio
using Optimisers
using ChainRulesCore
import ChainRulesCore: rrule
using Flux
using Zygote

function my_dense!(x, w, y, ∇x, ∇w)
    @tullio y[h, b] = x[f, b] * w[h, f]
    return y
end
function rrule(::typeof(my_dense!), x, w, y, ∇x, ∇w)
    y = my_dense!(x, w, y, ∇x, ∇w)
    my_dense!_pullback(ȳ) = (
        NoTangent(),
        my_dense!_Δx(∇x, unthunk(ȳ), w),
        my_dense!_Δw(∇w, unthunk(ȳ), x),
        NoTangent(),
        NoTangent(),
        NoTangent(),
    )
    return y, my_dense!_pullback
end
function my_dense!_Δx(∇x, ȳ, w)
    @tullio ∇x[f, b] = w[h, f] * ȳ[h, b]
end
function my_dense!_Δw(∇w, ȳ, x)
    @tullio ∇w[h, f] = x[f, b] * ȳ[h, b]
end

# initialize data
seed!(123)
b = 256 # batch size
f = 64 # in dim
h = 32 # out dim
x = rand(f, b)
w = rand(h, f)
y = rand(h, b)
∇x = similar(x)
∇w = similar(w)

@time out = my_dense!(x, w, y, ∇x, ∇w);
myloss(x, w, y, ∇x, ∇w) = mean(my_dense!(x, w, y, ∇x, ∇w))
myloss(x, w, y, ∇x, ∇w)

# 12.300 μs (18 allocations: 992 bytes)
@btime myloss($x, $w, $y, $∇x, $∇w);
# 17.800 μs (17 allocations: 960 bytes)
@btime my_dense!_Δx($∇x, $y, $w);
# 10.300 μs (17 allocations: 960 bytes)
@btime my_dense!_Δw($∇w, $y, $x);

function myfitA(x, ps, y, ∇x, ∇w)
    grads = Zygote.gradient(ps -> myloss(x, ps[:w], y, ∇x, ∇w), ps)
    return grads
end
ps = Dict(:w => w)
@time mygrads = myfitA(x, ps, y, ∇x, ∇w);
mygrads[1][:w]

# 62.100 μs (83 allocations: 134.69 KiB)
@btime mygrads = myfitA($x, $ps, $y, $∇x, $∇w);


######################################
# Allocating Non Mutating variation
######################################
function my_denseB!(x, w)
    @tullio y[h, b] := x[f, b] * w[h, f]
    return y
end

# initialize data
seed!(123)
b = 2048 # batch size
f = 128 # in dim
h = 128 # out dim
x = rand(f, b)
w = rand(h, f)

@time out = my_denseB!(x, w);
mylossB(x, w) = mean(my_denseB!(x, w))
mylossB(x, w)

# 13.700 μs (20 allocations: 65.02 KiB)
@btime mylossB($x, $w);
function myfitB(x, ps)
    grads = Zygote.gradient(ps -> mylossB(x, ps[:w]), ps)
    return grads
end
ps = Dict(:w => w)
@time mygradsB = myfitB(x, ps);
mygradsB[1][:w];

# 91.200 μs (99 allocations: 341.44 KiB)
@btime mygradsB = myfitB($x, $ps);

###########################
# Flux reference
###########################
m = Dense(w, false)
myloss0(x) = mean(m(x))
myloss0(x)

# 107.400 μs (5 allocations: 128.11 KiB)
@btime myloss0($x);
@btime $w * $x;

function fit0(x, loss, ps)
    grads = Zygote.gradient(() -> loss(x), ps)
    return grads
end
ps = Flux.params(m)
@time mygrads0 = fit0(x, myloss0, ps);
mygrads0[ps[1]]

# 335.000 μs (46 allocations: 340.69 KiB)
@btime mygrads0 = fit0($x, $myloss0, $ps);
