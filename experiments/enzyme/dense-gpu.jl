using Revise
# using LoopVectorization
# using Tullio
using Random: seed!
using Base.Threads
using Enzyme
using CUDA
using BenchmarkTools

function f3(x::AbstractArray, w::AbstractArray)
    z = sum(w * x)
    return z
end;

seed!(123)
bs = 128
f = 16
h1 = 32
# bs = 4096
# f = 256
# h1 = 512
w1 = CUDA.randn(h1, f) .* 0.01f0;
b1 = CUDA.randn(h1);
x = CUDA.randn(f, bs) .* 0.01f0;

dw1 = CUDA.zeros(h1, f);
db1 = CUDA.zeros(h1);
dx = CUDA.zeros(f, bs);

Duplicated(w1, dw1)

f3(x, w1)

# ERROR: Conversion of boxed type Matrix{Float64} is not allowed
# Array needs to be duplicated - not just Active
# y2 = Enzyme.autodiff(Reverse, f3, x, Active(w1), Active(b1));
out = Enzyme.autodiff(Reverse, f3, x, Duplicated(w1, dw1));

@time out = Enzyme.autodiff(Reverse, f3, Const(x), Duplicated(w1, dw1))
out = Enzyme.autodiff(Reverse, f3, Const, x, Duplicated(w1, dw1));
out = Enzyme.autodiff(Reverse, f3, Active, x, Duplicated(w1, dw1));

@btime Enzyme.autodiff(Reverse, f3, Const($x), Duplicated($w1, $dw1));

# y2 = Enzyme.autodiff(Reverse, f3, Duplicated(x, bx), Duplicated(w1, dw1), Duplicated(b1, db1));

function f4(x::Array{Float64}, w::Array{Float64})
    z = w * x
    return sum(z)
end;

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
dw1 = zeros(h1, f);
x = randn(f, bs) .* 0.01;

@time f4(x, w1)
# @btime f4($x, $w1)
@time out = Enzyme.autodiff(Reverse, f4, Const, Const(x), Duplicated(w1, dw1));

# ERROR: Conversion of boxed type Matrix{Float64} is not allowed
# Array needs to be duplicated - not just Active
# y2 = Enzyme.autodiff(Reverse, f3, x, Active(w1), Active(b1));
@time Enzyme.autodiff(Reverse, f4, Const(x), Duplicated(w1, dw1));
# @time Enzyme.autodiff(Reverse, f4, Const($x), Duplicated($w1, $dw1));


using Base.Experimental: @aliasscope, Const
function f5(x::Array{Float64}, w::Array{Float64})
    @aliasscope let x = x, w = w
        sum(w * x)
    end
    return nothing
end;

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
dw1 = zeros(h1, f);
x = randn(f, bs) .* 0.01;

@time f5(x, w1)
# @btime f5($x, $w1)
@time out = Enzyme.autodiff(Reverse, f5, Const, Const(x), Duplicated(w1, dw1))


using Base.Experimental: @aliasscope, Const
function f6(x::Array{Float64}, w::Array{Float64})
    x = Base.unalias(x, x)
    w = Base.unalias(w, w)
    z = sum(w * x)
    return z
end;

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
dw1 = zeros(h1, f);
x = randn(f, bs) .* 0.01;

@time f6(x, w1)
# @btime f6($x, $w1)
@time out = Enzyme.autodiff(Reverse, f6, Const, Const(x), Duplicated(w1, dw1))


############################################
# Custom mat-mul
############################################
function mymul!(R, A, B)
    @assert axes(A,2) == axes(B,1)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end
    @inbounds for j in axes(B, 2), i in axes(A, 1)
        @inbounds @simd for k in axes(A,2)
            R[i,j] += A[i,k] * B[k,j]
        end
    end
    nothing
end

Random.seed!(1234)
A = rand(512, 256);
B = rand(256, 4096);

R = zeros(size(A,1), size(B,2))
∂z_∂R = rand(size(R)...)  # Some gradient/tangent passed to us
∂z_∂R0 = copyto!(similar(∂z_∂R), ∂z_∂R)  # exact copy for comparison

∂z_∂A = zero(A)
∂z_∂B = zero(B)

@time mymul!(R, A, B)
@time R_ = A * B;

Enzyme.autodiff(mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))
@time Enzyme.autodiff(mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))
