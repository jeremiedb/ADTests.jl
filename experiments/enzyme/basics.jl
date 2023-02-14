using Revise
# using LoopVectorization
# using Tullio
using Base.Threads 
using Enzyme

function f1(x::Array{Float64}, y::Array{Float64})
    y[1] = x[1] * x[1] + x[2] * x[1]
    return nothing
end;

x  = [3.0, 5.0]
bx = [0.0, 0.0]
y  = [0.0]
by = [1.0];

bx
by
Enzyme.autodiff(f1, Duplicated(x, bx), Duplicated(y, by));
Enzyme.autodiff(Reverse, f1, Duplicated(x, bx), Duplicated(y, by));


function f2(x::Array{Float64})
    y = x[1] * x[1] + x[2] * x[1]
    return y
end;

x  = [3.0, 5.0]
bx = [0.0, 0.0]

bx
by
Enzyme.autodiff(Reverse, f2, Duplicated(x, bx));


function f3(x::Array{Float64}, y::Array{Float64})
    @threads for i in eachindex(x)
        y[i] = x[1] * x[1] + x[2] * x[1]
    end
    return nothing
end;

x  = [3.0, 5.0]
bx = [0.0, 0.0]
y  = [0.0, 0.0]
by = [1.0, 1.0];

Enzyme.autodiff(Reverse, f3, Duplicated(x, bx), Duplicated(y, by));
# f3(x, y)


