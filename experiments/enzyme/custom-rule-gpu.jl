using Revise
using CUDA
using Enzyme
using ChainRulesCore
import ChainRulesCore: rrule
using Zygote
# import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule, has_rrule_from_sig
# using .EnzymeRules

function f_gpu(x)
    threads = size(x, 2)
    @cuda blocks = (1,) threads = threads f_kernel!(x)
    CUDA.synchronize()
end

function f_kernel!(x)
    j = threadIdx().x
    for i = 2:1:size(x, 1)
        x[i, j] += x[i-1, j]
    end
    sync_threads()
    return nothing
end

x = CUDA.rand(5, 3)
dx = CUDA.zeros(5, 3)
f_gpu(x)

Enzyme.autodiff(Reverse, f_gpu, Duplicated(x, dx))

########################
# tak1 1
########################
function square_gpu!(x)
    @cuda blocks = (1,) threads = length(x) square_kernel!(x)
    CUDA.synchronize()
    return x
end
function square_kernel!(x)
    i = threadIdx().x
    x[i] = x[i]^2
    sync_threads()
    return nothing
end

# current Enzyme syntax to diff CUDA kernels
function square_kernel_grad!(x, dx)
    autodiff_deferred(Reverse, square_kernel!, Duplicated(x, dx))
    return nothing
end

x = CUDA.zeros(3, 2) .+ CuArray(1:2:5)
dx = CUDA.ones(3, 2)
@cuda blocks = (1,) threads = length(x) square_kernel_grad!(x, dx);

function my_model(x)
    x = square_gpu!(x)
    return sum(x)
end

function rrule(::typeof(square_gpu!), x)
    _x = copy(x)
    x = square_gpu!(x)
    square_gpu!_pullback(ȳ) = NoTangent(), ∇_square_kernel(_x, unthunk(ȳ))
    return x, square_gpu!_pullback
end
function ∇_square_kernel(x, ȳ)
    @cuda threads = length(x) square_kernel_grad!(x, ȳ)
    CUDA.synchronize()
    return ȳ
end

x1 = CUDA.zeros(3, 2) .+ CuArray(1:3)
# grads = Zygote.gradient(x -> my_model(x), x1)[1]
val, grads = Zygote.withgradient(x -> my_model(x), x1)
# my_model(x1)

########################
# take 2
########################
function square_gpu(x)
    out = similar(x)
    out .= x
    @cuda blocks = (1,) threads = length(out) square_kernel!(out)
    CUDA.synchronize()
    return out
end
function square_kernel!(x)
    i = threadIdx().x
    x[i] = x[i]^2
    sync_threads()
    return nothing
end

function my_model_2(x)
    out = square_gpu(x)
    return sum(out)
end

function rrule(::typeof(square_gpu), x)
    out = square_gpu(x)
    square_gpu_pullback(ȳ) = NoTangent(), ∇_square_kernel(x, unthunk(ȳ))
    return out, square_gpu_pullback
end
function ∇_square_kernel(x, ȳ)
    @cuda threads = length(x) square_kernel_grad!(x, ȳ)
    CUDA.synchronize()
    return ȳ
end

x1 = CUDA.zeros(3, 2) .+ CuArray(1:3);
out = square_gpu(x1)
my_model_2(x1)
# grads = Zygote.gradient(x -> my_model(x), x1)[1]
val, grads = Zygote.withgradient(x -> my_model_2(x), x1)


########################
# take 3 - split mode
########################
function square_gpu(x)
    out = similar(x)
    out .= x
    @cuda blocks = (1,) threads = length(out) square_kernel!(out)
    CUDA.synchronize()
    return out
end
function square_kernel!(x)
    i = threadIdx().x
    x[i] = x[i]^2
    sync_threads()
    return nothing
end
function my_model_2(x)
    out = square_gpu(x)
    return sum(out)
end

function square_kernel_split!(x, dx)
    autodiff_deferred(ReverseWithPrimal, square_kernel!, Duplicated(x, dx))
    return nothing
end

x = CUDA.zeros(3, 2) .+ CuArray(1:3)
dx = CUDA.ones(3, 2)
square_gpu(x)
forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, square_gpu, Duplicated, Duplicated{typeof(x)})
forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, square_kernel!, Const, Duplicated{typeof(x)})

A = [2.2]; ∂A = zero(A)
v = 3.3
function f(A, v)
    res = A[1] * v
    A[1] = 0
    res
end
forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, f, Active, Duplicated{typeof(A)}, Active{typeof(v)})
tape, result, shadow_result  = forward(Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Duplicated(A, ∂A), Active(v), 1.0, tape)[1]

function rrule(::typeof(square_gpu), x)
    out = square_gpu(x)
    square_gpu_pullback(ȳ) = NoTangent(), ∇_square_kernel(x, unthunk(ȳ))
    return out, square_gpu_pullback
end
function ∇_square_kernel(x, ȳ)
    @cuda threads = length(x) square_kernel_grad!(x, ȳ)
    CUDA.synchronize()
    return ȳ
end

x1 = CUDA.zeros(3, 2) .+ CuArray(1:3);
out = square_gpu(x1)
my_model_2(x1)
# grads = Zygote.gradient(x -> my_model(x), x1)[1]
val, grads = Zygote.withgradient(x -> my_model_2(x), x1)

forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, f, Active, Duplicated{typeof(A)}, Active{typeof(v)})
