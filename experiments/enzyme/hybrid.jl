using Revise
using Random: seed!
using Base.Threads
using Enzyme
using BenchmarkTools

function mutate_fun!(z::AbstractArray)
    for i in eachindex(z)
        z[i] .+ z[i]^2
    end
    return nothing
end;

function hybrid(x::Array{Float64}, w::Array{Float64})
    z = w * x
    mutate_fun!(z)
    return sum(z)
end

seed!(123)
bs = 4096
f = 256
h1 = 512
x = randn(f, bs) .* 0.01f0;
dx = zeros(f, bs);
w1 = randn(h1, f) .* 0.01f0;
dw1 = zeros(h1, f);

@time hybrid(x, w1)
@time out = Enzyme.autodiff(Reverse, hybrid, x, Duplicated(w1, dw1));
# y2 = Enzyme.autodiff(Reverse, f3, Duplicated(x, bx), Duplicated(w1, dw1), Duplicated(b1, db1));
