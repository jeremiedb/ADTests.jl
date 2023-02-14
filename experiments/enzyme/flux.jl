using Revise
# using LoopVectorization
# using Tullio
using Random: seed!
using Base.Threads
using Flux
using Enzyme
using BenchmarkTools

using Flux, Random, Enzyme
rng = Random.default_rng()

loss(model, x) = sum(model(x))

model = Chain(Dense(2 => 4), BatchNorm(4), Dense(4 => 2))
x = randn(rng, Float32, 2, 1)

dmodel = Flux.fmap(model) do x
    x isa Array ? zero(x) : x
end

Enzyme.autodiff(loss, Duplicated(model, dmodel), Const(x))
println(dmodel)


model = Chain(Dense(256 => 512))
x = randn(rng, Float32, 256, 4096);
loss(model, x) = sum(model(x))
dmodel = Flux.fmap(model) do x
    x isa Array ? zero(x) : x
end
@time Enzyme.autodiff(loss, Duplicated(model, dmodel), Const(x))

println(dmodel)

function f1(x::Array{Float64}, y::Array{Float64})
    y[1] = sum(x * x)
    return nothing
end;
