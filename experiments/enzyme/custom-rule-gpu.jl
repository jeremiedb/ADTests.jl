using Revise
using CUDA
using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule, has_rrule_from_sig
using .EnzymeRules

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
# 
########################
function sqrt_gpu(x)
    threads = size(x)
    @cuda blocks = (1,) threads = threads sqrt_kernel!(x)
    CUDA.synchronize()
end

function sqrt_kernel!(x)
    i, j = threadIdx().x, threadIdx().y
    x[i, j] = sqrt(x[i, j])
    sync_threads()
    return nothing
end

x = CUDA.zeros(5, 3) .+ CuArray(1:2:9)
dx = CUDA.ones(5, 3)
Enzyme.autodiff(Reverse, sqrt_gpu, Duplicated(x, dx))
