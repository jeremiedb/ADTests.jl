using Revise
using BenchmarkTools
using Statistics
using Random: seed!
using LoopVectorization
using Base.Threads: @threads
using LinearAlgebra: mul!
# using StaticArrays
using Tullio
# using Optimisers

################################
# rnn debug
################################
using Flux
using Flux: Losses
nsteps = 4
xs = [rand(Float32, 3, 2) for i = 1:nsteps]
m = Chain(RNN(3, 5))
y = [m(x) for x in xs]
# m = Chain(RNN(3, 5), Dense(5, 1), vec)

m = Chain(RNN(3, 5), RNN(5, 3))
y = [m(x) for x in xs]
y[1]

xn = Vector{Matrix{Float32}}()
push!(xn, m(x[1]))
for i in 2:nsteps
    push!(xn, m(xn[i-1]))
end
y = Flux.stack(xn; dims=2)

function loss(m, x)
    xn = Vector{Matrix{Float32}}()
    push!(xn, m(x[1]))
    for i in 2:nsteps
        push!(xn, m(xn[i-1]))
    end
    y = Flux.stack(xn; dims=2)
    return sum(y)
end

using Zygote
grads = gradient(model -> loss(model, x), m)

function loss(m, x)
    xn = Vector{Matrix{Float32}}()
    push!(xn, m(x[1]))
    for i in 2:nsteps
        push!(xn, m(xn[i-1]))
    end
    y = Flux.stack(xn; dims=2)
    return sum(y)
end

################################
# vect of mat-vec mul experiment
################################
bs = 2048
h1 = 768
h2 = 1024
w = randn(h2, h1) .* 0.1;
x = randn(h1, bs) .* 0.1;
b = [x[:, i] for i = 1:bs];
b[1]
x[:, 1]

# 17.285 ms (2 allocations: 16.00 MiB)
@btime $w * $x;
m0 = w * x;

function mul_1(b::Vector{Vector{T}}, w::Matrix{T}) where {T}
    # y = [Vector{Float64}(undef, size(w,1)) for _ in eachindex(b)]
    y = [zeros(Float64, size(w, 1)) for _ in eachindex(b)]
    @threads for i in eachindex(b)
        # @inbounds y[i] .= w * b[i]
        # @turbo y[i] .= w * b[i]
        mul!(y[i], w, b[i])
    end
    return y
end

# 236.109 ms (4147 allocations: 32.52 MiB)
@btime mul_1($b, $w);
# @code_warntype mul_1(b, w);

y = [Vector{Float64}(undef, size(w, 1)) for _ in eachindex(b)]
y[1] = w * b[1]

y[1] .= 0
@tturbo y[1] = w * b[1];
@time @turbo w * b[1];

m1 = mul_1(b, w);
all(m0[:, 1] .== m1[1])
maximum(abs.(m0[:, 1] .- m1[1]))

################################
# end experiment
################################

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

