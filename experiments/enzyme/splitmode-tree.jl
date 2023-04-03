using Revise
using CUDA
using Enzyme
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
using Base.Threads
# import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule, has_rrule_from_sig
# using .EnzymeRules

#############################################
# tree
#############################################
function cnw!(x, y)
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

# cnw!(x, y)
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    Const{typeof(cnw!)},
    Const,
    Duplicated{typeof(x)},
    Duplicated{typeof(y)},
)
tape, result, shadow_result = forward(Const(cnw!), Duplicated(x, ∂x), Duplicated(y, ∂y))
grads = backward(Const(cnw!), Duplicated(x, ∂x), Duplicated(y, ∂y), tape)




#############################################
# tree
#############################################
function lw!(nw, cw, lw)
    i = 2
    stop = size(cw, 1)
    while i < stop
        cw[i] = cw[i>>1] * nw[i>>1]
        cw[i+1] = cw[i>>1] * (1 - nw[i>>1])
        i +=2
    end
    k = stop
    stop += size(lw, 1)
    while i < stop
        lw[i-k] = cw[i>>1] * nw[i>>1]
        lw[i-k+1] = cw[i>>1] * (1 - nw[i>>1])
        i +=2
    end
    return nothing
end

nw = [0.6, 0.2, 0.8];
∂nw = zeros(3);

cw = ones(3);
∂cw = zeros(3);

lw = zeros(4);
# ∂lw = [-0.1, 0.0, 0.2, 1.0];
# ȳ = deepcopy(∂lw);
∂lw = zeros(4);
ȳ = [-0.1, 0.0, 0.2, 1.0];

# lw!(nw, cw, lw)
forward, backward = autodiff_thunk(
    ReverseSplitNoPrimal,
    # ReverseSplitWithPrimal,
    Const{typeof(lw!)},
    Const,
    Duplicated{typeof(nw)},
    Duplicated{typeof(cw)},
    # Const{typeof(lw)},
    Duplicated{typeof(lw)},
)
tape, result, shadow_result = forward(Const(lw!), Duplicated(nw, ∂nw), Duplicated(cw, ∂cw), Duplicated(lw, ∂lw))
_ = backward(Const(lw!), Duplicated(nw, ∂nw), Duplicated(cw, ∂cw), Duplicated(lw, Array(ȳ)), tape)


function leaf_weights_enz(nw)
    cw = ones(eltype(nw), size(nw))
    lw = zeros(eltype(nw), size(nw, 1) + 1, size(nw, 2), size(nw, 3))
    leaf_weights_enz!(nw, cw, lw)
    return lw
end

function rrule(::typeof(leaf_weights_enz), nw)

    cw = ones(eltype(nw), size(nw))
    lw = zeros(eltype(nw), size(nw, 1) + 1, size(nw, 2), size(nw, 3))
    # project_lw = ProjectTo(lw)

    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal,
        Const{typeof(leaf_weights_enz!)},
        Const,
        Duplicated{typeof(nw)},
        Duplicated{typeof(cw)},
        Duplicated{typeof(lw)},
    )

    ∂nw = zero(nw)
    ∂cw = zero(cw)
    ∂lw = zero(lw)

    tape, result, shadow_result = forward(
        Const(leaf_weights_enz!),
        Duplicated(nw, ∂nw),
        Duplicated(cw, ∂cw),
        Duplicated(lw, ∂lw),
    )

    function lw_enz_pullback(ȳ)
        @info "minimum(ȳ): " minimum(ȳ)
        @info "maximum(ȳ): " maximum(ȳ)
        @info "minimum(cw): " minimum(cw)
        @info "maximum(cw): " maximum(cw)
        @info "minimum(lw): " minimum(lw)
        @info "maximum(lw): " maximum(lw)
        grad = reverse(
            Const(leaf_weights_enz!),
            Duplicated(nw, ∂nw),
            Duplicated(cw, ∂cw),
            Duplicated(lw, Array(ȳ)),
            tape,
        )
        @info "minimum(∂nw): " minimum(∂nw)
        @info "maximum(∂nw): " maximum(∂nw)
        return NoTangent(), ∂nw
    end
    return lw, lw_enz_pullback
end