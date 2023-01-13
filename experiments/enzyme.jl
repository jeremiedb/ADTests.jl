using Revise
using LoopVectorization
using Tullio
using Optimisers
using Enzyme


f1(x) = x * x
# Returns a tuple of active returns, which in this case is simply (2.0,)
autodiff(Reverse, f1, Active(1.0))
autodiff(Reverse, f1, Active(3.0))

f2(x, y) = x * y
# Returns a tuple of active returns, which in this case is simply (2.0,)
autodiff(Reverse, f2, Active(3.0), Active(5.0))

x = rand(2,3)
y = rand(3,5)
# autodiff(Reverse, f2, Active(x), y)
# autodiff(Reverse, f2, Active([2.0, 3.0]), Active([3.0, 5.0]))

function f2(x, tmp, n)
    tmp[1] = 1
    for i = 1:n
        tmp[1] *= x
    end
    tmp[1]
end
# Incorrect [ returns (0.0,) ]
Enzyme.autodiff(f2, Active(1.2), Const(Vector{Float64}(undef, 1)), Const(5))
Enzyme.autodiff(
    f2,
    Active(1.2),
    Duplicated(Vector{Float64}(undef, 1), Vector{Float64}(undef, 1)),
    Const(5),
)
