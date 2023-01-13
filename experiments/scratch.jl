using Revise
using BenchmarkTools
using Statistics
using Random: seed!
using LoopVectorization
using Tullio
using Optimisers


function my_dense!(x, w, y)
    @tullio y[h, b] = x[f, b] * w[h, f]
    return y
end
function my_dense!_pullback(ȳ)
    my_dense!_Δx(∇x, unthunk(ȳ), w),
    my_dense!_Δw(∇w, unthunk(ȳ), x),
    NoTangent(),
    NoTangent(),
    NoTangent(),
    return y, my_dense!_pullback
end

# layer_f
# layer_b

