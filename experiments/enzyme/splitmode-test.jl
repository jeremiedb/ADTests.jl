using Revise
using CUDA
using Enzyme
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
using Base.Threads
# import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule, has_rrule_from_sig
# using .EnzymeRules


#####################################################
# docs base exasmple
#####################################################
A = [2.2];
∂A = zero(A);
v = 3.3
function f0(A, v)
    res = A[1] * v
    A[1] = 0
    res
end
function f1(A, v)
    res = A[1] * v
    A[1] = 0
    res
end
forward, reverse = autodiff_thunk(
    ReverseSplitWithPrimal,
    Const{typeof(f1)},
    Active,
    Duplicated{typeof(A)},
    Active{typeof(v)},
)
tape, result, shadow_result = forward(Const(f1), Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Const(f1), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]


#####################################################
# docs base exasmple - implement with Zygote rrule
#####################################################
function rrule(::typeof(f1), A, v)
    ∂A = zero(A)
    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal,
        Const{typeof(f1)},
        Active,
        Duplicated{typeof(A)},
        Active{typeof(v)},
    )
    tape, result, shadow_result = forward(Const(f1), Duplicated(A, ∂A), Active(v))

    pullback(ȳ) =
        NoTangent(), reverse(Const(f1), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]...
    return result, pullback
end

grad = Zygote.gradient(A, v) do A, v
    f0(A, v)
end
grad = Zygote.gradient(A, v) do A, v
    f1(A, v)
end
val, grad = Zygote.withgradient(A, v) do A, v
    f1(A, v)
end


#############################################
# vector mutation
#############################################
function f2(x)
    for i in eachindex(x)
        x[i] = x[i]^2
    end
    return nothing
end

x = zeros(5) .+ collect(1:5);
∂x = ones(5);
# grad = autodiff(Reverse, f1, Duplicated(x, ∂x))
# grad, val = autodiff(ReverseWithPrimal, f1, Const, Duplicated(x, ∂x))

forward, backward =
    autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(f2)}, Const, Duplicated{typeof(x)})
tape, result, shadow_result = forward(Const(f2), Duplicated(x, ∂x))
# reverse(Duplicated(x, ∂x), ones(5), tape)
# _, ∂v = reverse(Duplicated(x, ∂x), ones(5), tape)[1]
# _, ∂v = reverse(Duplicated(x, ∂x), 1.0, tape)[1]
grads = backward(Const(f2), Duplicated(x, ∂x), tape)
grads[1]

#############################################
# vector mutation - Zygote
#############################################
function rrule(::typeof(f2), x)
    ∂x = zero(x) .+ 1
    forward, backward = autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(f2)},
        Const,
        Duplicated{typeof(x)},
    )

    tape, result, shadow_result = forward(Const(f2), Duplicated(x, ∂x))

    function pullback(ȳ)
        _ = backward(Const(f2), Duplicated(x, ∂x), tape)[1][1]
        return NoTangent(), ȳ .* ∂x
    end
    return x, pullback

end

x = zeros(5) .+ collect(1:5);
grad = Zygote.gradient(x) do x
    3 * sum(f2(x))
end
val, grad = Zygote.withgradient(x) do x
    sum(f2(x))
end


#############################################
# 2-vectors mutation
#############################################
function f3(x, y)
    for i in eachindex(x)
        for j = i:length(y)
            y[j] *= x[i]
        end
    end
    return nothing
end

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);
y = ones(3);
∂y = ones(3);
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f3)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
)
tape, result, shadow_result = forward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y))
grads = backward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)

#############################################
# vector mutation - Zygote
#############################################
function rrule(::typeof(f3), x, y)
    ∂x = zero(x) .+ 1
    forward, backward = autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(f2)},
        Const,
        Duplicated{typeof(x)},
    )

    tape, result, shadow_result = forward(Const(f2), Duplicated(x, ∂x))

    function pullback(ȳ)
        _ = backward(Const(f2), Duplicated(x, ∂x), tape)[1][1]
        return NoTangent(), ∂x
    end
    return x, pullback

end

x = zeros(5) .+ collect(1:5);
grad = Zygote.gradient(x) do x
    sum(f2(x))
end
val, grad = Zygote.withgradient(x) do x
    sum(f2(x))
end



#############################################
# tree
#############################################
function cnw!(x)
    for i in eachindex(x)
        for j = i:length(y)
            y[j] *= x[i]
        end
    end
    return nothing
end

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);

y = ones(3);
∂y = ones(3);

forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f3)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
)
tape, result, shadow_result = forward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y))
grads = backward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)
