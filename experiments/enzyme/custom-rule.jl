using Revise
using Enzyme
import .EnzymeRules: augmented_primal, reverse
using .EnzymeRules

function my_sqrt_scalar(x)
    return sqrt(x)
end

function augmented_primal(
    config::ConfigWidth{1},
    func::Const{typeof(my_sqrt_scalar)},
    ::Type{<:Active},
    x::Active)

    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(
    config::Config,
    ::Const{typeof(my_sqrt_scalar)},
    dret::Active,
    tape,
    x::Active)

    if needs_primal(config)
        # return (x.val*dret.val,)
        return (23.0,)
    else
        # return (100+2*x.val*dret.val,)
        return (43.0,)
    end
end

x = 9.0
Enzyme.autodiff(Reverse, my_sqrt_scalar, Const, Active(x))
Enzyme.autodiff(Reverse, my_sqrt_scalar, Active, Active(x))

function my_sqrt(x)
    for i in eachindex(x)
        x[i] = sqrt(x[i])
    end
    return nothing
end

# function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(my_sqrt)}, ::Type{<:Active}, x::Active)
#     if needs_primal(config)
#         return AugmentedReturn(func.val(x.val), nothing, nothing)
#     else
#         return AugmentedReturn(nothing, nothing, nothing)
#     end
# end

# function reverse(config, ::Const{typeof(my_sqrt)}, dret::Duplicated, tape, x::Duplicated)
#     if needs_primal(config)
#         return (10+2*x.val*dret.val,)
#     else
#         @info typeof(x)
#         return (100+2*x.val*dret.val,)
#     end
# end
x = zeros(3) .+ [1, 4, 9]
dx = ones(3)
Enzyme.autodiff(Reverse, my_sqrt, Duplicated(x, dx))
1 ./ 2 ./ sqrt.(x)
