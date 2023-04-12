using Revise
using CUDA
using Enzyme
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
using Base.Threads

#############################################
# vector mutation
# works fine
#############################################
function f1(x)
    for i in eachindex(x)
        x[i] = x[i]^2
    end
    return nothing
end

x = zeros(5) .+ collect(1:5);
∂x = ones(5);
_, val = autodiff(ReverseWithPrimal, f1, Duplicated(x, ∂x))

forward, backward =
    autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f1)}, Const, Duplicated{typeof(x)})
tape, result, shadow_result = forward(Const(f1), Duplicated(x, ∂x))
_ = backward(Const(f1), Duplicated(x, ∂x), tape)


#############################################
# vector mutation - threads
# works fine
#############################################
function f1t(x)
    @threads for i in eachindex(x)
        x[i] = x[i]^2
    end
    return nothing
end

x = zeros(5) .+ collect(1:5);
∂x = ones(5);
_, val = autodiff(ReverseWithPrimal, f1t, Duplicated(x, ∂x))

x = zeros(5) .+ collect(1:5);
∂x = ones(5);
forward, backward =
    autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f1t)}, Const, Duplicated{typeof(x)})
tape, result, shadow_result = forward(Const(f1t), Duplicated(x, ∂x))
_ = backward(Const(f1t), Duplicated(x, ∂x), tape)


#############################################
# 2-vectors mutation
#############################################
function f2(x, y)
    for i in eachindex(x)
        y[i] *= x[i]^2
    end
    return nothing
end

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);
y = ones(3);
∂y = ones(3);
_ = autodiff(Reverse, f2, Duplicated(x, ∂x), Duplicated(y, ∂y))

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);
y = ones(3);
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f2)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
);
tape, result, shadow_result = forward(Const(f2), Duplicated(x, ∂x), Duplicated(y, ∂y));
∂y = ones(3);
_ = backward(Const(f2), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)

#############################################
# 2-vectors mutation
#############################################
function f2t(x, y)
    @threads for i in eachindex(x)
        y[i] *= x[i]^2
    end
    return nothing
end

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);
y = ones(3);
∂y = ones(3);
_ = autodiff(Reverse, f2t, Duplicated(x, ∂x), Duplicated(y, ∂y))

x = zeros(3) .+ collect(2:4);
∂x = zeros(3);
y = ones(3);
∂y = zeros(3);
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f2t)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
);
tape, result, shadow_result = forward(Const(f2t), Duplicated(x, ∂x), Duplicated(y, ∂y));
∂y = ones(3);
_ = backward(Const(f2t), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)



########################################
# double loops
########################################
function f3(x, y)
    for j in axes(x, 2)
        for i = axes(x, 1)
            y[i, j] *= x[i, j]^2
        end
    end
    return nothing
end

# x = zeros(3,4) .+ collect(2:4);
# ∂x = zeros(3,4);
# y = ones(3,4);
# ∂y = ones(3,4);
# _ = autodiff(Reverse, f3, Duplicated(x, ∂x), Duplicated(y, ∂y))

x = zeros(3,4) .+ collect(2:4);
∂x = zeros(3,4);
y = ones(3,4);
∂y = zeros(3,4);
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f3)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
);
tape, result, shadow_result = forward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y));
∂y = ones(3,4);
_ = backward(Const(f3), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)

############################################
# double loops - 
# same ∂y in forward and reverse required
############################################
function f3t(x, y)
    @threads for j in axes(x, 2)
        for i = axes(x, 1)
            y[i, j] *= x[i, j]^2
        end
    end
    return nothing
end

x = zeros(3,4) .+ collect(2:4);
∂x = zeros(3,4);
y = ones(3,4);
∂y = zeros(3,4);
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(f3t)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
);
tape, result, shadow_result = forward(Const(f3t), Duplicated(x, ∂x), Duplicated(y, ∂y));
∂y .= ones(3,4);
_ = backward(Const(f3t), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)
