using Revise
# using LoopVectorization
# using Tullio
using Random: seed!
using Base.Threads 
using Enzyme

using BenchmarkTools

function f1(x::Array{Float64}, y::Array{Float64})
    y[1] = sum(x * x)
    return nothing
end;

x  = rand(3, 3)
bx = zeros(3, 3)
y  = [0.0]
by = [1.0];

bx
by
Enzyme.autodiff(f1, Const, Duplicated(x, bx), Duplicated(y, by))
Enzyme.autodiff(Reverse, f1, Duplicated(x, bx), Duplicated(y, by));

function f2(x::Array{Float64})
    y = sum(x * x)
    return y
end;

x  = rand(3, 3)
bx = zeros(3, 3)
y  = 0.0;
by = 1.0;

f2(x)
y2 = Enzyme.autodiff(Reverse, f2, Duplicated(x, bx));


function f3(x::Array{Float64}, w::Array{Float64}, b::Vector{Float64})
    # z = sum(exp.(w * x .+ b))
    z = sum(w * x)
    return z
end;

seed!(123)
# bs = 128
# f = 16
# h1 = 32
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

dw1 = zeros(h1, f);
db1 = zeros(h1);
dx = zeros(f, bs);

f3(x, w1, b1)

# ERROR: Conversion of boxed type Matrix{Float64} is not allowed
# Array needs to be duplicated - not just Active
# y2 = Enzyme.autodiff(Reverse, f3, x, Active(w1), Active(b1));
@time out = Enzyme.autodiff(Reverse, f3, Const(x), Duplicated(w1, dw1), Duplicated(b1, db1))
out = Enzyme.autodiff(Reverse, f3, Const, x, Duplicated(w1, dw1), Duplicated(b1, db1));
out = Enzyme.autodiff(Reverse, f3, Active, x, Duplicated(w1, dw1), Duplicated(b1, db1));

@btime Enzyme.autodiff(Reverse, f3, Const($x), Duplicated($w1, $dw1), Duplicated($b1, $db1));

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
@btime f4($x, $w1)
@time out = Enzyme.autodiff(Reverse, f4, Const, Const(x), Duplicated(w1, dw1))

# ERROR: Conversion of boxed type Matrix{Float64} is not allowed
# Array needs to be duplicated - not just Active
# y2 = Enzyme.autodiff(Reverse, f3, x, Active(w1), Active(b1));
@btime Enzyme.autodiff(Reverse, f4, Const($x), Duplicated($w1, $dw1));
