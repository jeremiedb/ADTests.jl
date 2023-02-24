using Enzyme
using Enzyme: EnzymeRules
# using .EnzymeRules: Config

function my_sqrt_scalar(x)
    for i in eachindex(x)
        x[i] = sqrt(x)
    end
    return nothing
end

function reverse(config::Config, ::Const{typeof(my_sqrt_scalar)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

x = 9.0
@time Enzyme.autodiff(my_sqrt_scalar, Const, Active(x))


function my_sqrt(x)
    for i in eachindex(x)
        x[i] = sqrt(x)
    end
    return nothing
end

function reverse(config, ::Const{typeof(my_sqrt)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

x = randn(10)
dx = similar(x)
@time Enzyme.autodiff(my_sqrt, Const, Duplicated(x, dx))
