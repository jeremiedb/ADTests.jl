using Revise
using BenchmarkTools
using Statistics
using Random: seed!
using LoopVectorization
using Tullio
using CUDA
using NNlib
using NNlibCUDA
using BenchmarkTools

CUDA.allowscalar(false)

# width, height, channels, batchsize
x_c = randn(128, 128, 16, 16);
# width, height, d_in, d_out
w_c = randn(3, 3, 16, 32)
@time conv(x_c, w_c);
# 62.868 ms (80 allocations: 201.56 MiB)
@btime conv($x_c, $w_c);

x_g = CuArray(x_c);
# width, height, d_in, d_out
w_g = CuArray(w_c);
CUDA.@time y_g = conv(x_g, w_g);
# 13.612 ms (128 allocations: 6.27 KiB)
@btime CUDA.@sync conv($x_g, $w_g);

function kernel_1!(y, x, w)
    tix, tiy, tiz = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy, bdz = blockDim().x, blockDim().y, blockDim().z
    bix, biy, biz = blockIdx().x, blockIdx().y, blockIdx().z
    gdx, gdy, gdz = gridDim().x, gridDim().y, gridDim().z

    # d2 = tix
    # c_out = biy

    d2_bs = cld(size(y, 2), bdx)
    d1_bs = cld(size(y, 1), bdy)

    # d1 = cld(size(y, 2), bdy)

    # for d2 in axes(y, 2)
    # for d1 in axes(y, 1)
    for d2_iter = 1:d2_bs
        d2 = (tix - 1) * d2_bs + d2_iter
        if d2 <= size(y, 2)
            for d1_iter = 1:d1_bs
                d1 = (tiy - 1) * d1_bs + d1_iter
                if d1 <= size(y, 1)
                    for c_out in axes(w, 4)
                        res = 0
                        for c_in in axes(w, 3)
                            for k2 in axes(w, 2)
                                for k1 in axes(w, 1)
                                    @inbounds res +=
                                        x[d1+k1-1, d2+k2-1, c_in, bix] *
                                        w[k1, k2, c_in, c_out]
                                end
                            end
                        end
                        y[d1, d2, c_out, bix] = res
                        # y[d1, d2, c_out, bix] = sum(view(w, :, :, :, c_out))
                    end
                end
            end
        end
    end
end

function myconv_1(x, w)
    y = CUDA.zeros(size(x, 1) - 2, size(x, 1) - 2, size(w, 4), size(x, 4))
    kernel = @cuda launch = false kernel_1!(y, x, w)
    config = launch_configuration(kernel.fun)
    # threads = min(length(y), config.threads)
    # threads = (size(y, 2), 4)
    threads = (126, 4)
    # threads = (size(y, 2), fld(config.threads, size(y, 1)))
    # @info threads
    blocks = (last(size(x)), 1)
    # blocks = (last(size(x)), last(size(w)))
    CUDA.@sync begin
        kernel(y, x, w; threads, blocks)
    end
    return y
end

# flip weights to match conv implementation
w2 = w_g[end:-1:1, end:-1:1, :, :]
y = myconv_1(x_g, w2);
CUDA.@time y = myconv_1(x_g, w2);
@btime myconv_1($x_g, $w2);


function kernel_2!(y, x, w)
    tix, tiy, tiz = threadIdx().x, threadIdx().y, threadIdx().z
    bdx, bdy, bdz = blockDim().x, blockDim().y, blockDim().z
    bix, biy, biz = blockIdx().x, blockIdx().y, blockIdx().z
    gdx, gdy, gdz = gridDim().x, gridDim().y, gridDim().z

    cache = @cuDynamicSharedMem(Float32, (size(w, 1), size(w, 2), size(w, 3)))
    if tix == 1 && tiy == 1
        for i in eachindex(cache)
            cache[i] = w[i]
        end
    end
    sync_threads()

    d2 = tix
    d1_bs = cld(size(y, 1), bdy)

    # for d2 in axes(y, 2)
    # for d1 in axes(y, 1)
    for c_out in axes(w, 4)
        # view(cache, :, : , :) .= view(w, :, :, :, c_out)
        for d1_iter = 1:d1_bs
            d1 = (tiy - 1) * d1_bs + d1_iter
            if d1 <= size(y, 1)
                res = 0
                for c_in in axes(cache, 3)
                    for k2 in axes(cache, 2)
                        for k1 in axes(cache, 1)
                            res += x[d1+k1-1, d2+k2-1, c_in, bix] * cache[k1, k2, c_in]
                        end
                    end
                end
                y[d1, d2, c_out, bix] = res
                # y[d1, d2, c_out, bix] = sum(view(w, :, :, :, c_out))
            end
        end
    end
    # end
end

function myconv_2(x, w)
    y = CUDA.zeros(size(x, 1) - 2, size(x, 1) - 2, size(w, 4), size(x, 4))
    kernel = @cuda launch = false kernel_2!(y, x, w)
    config = launch_configuration(kernel.fun)
    # threads = min(length(y), config.threads)
    threads = (size(y, 2), 4)
    # threads = (size(y, 2), fld(config.threads, size(y, 1)))
    # @info threads
    blocks = last(size(x))
    CUDA.@sync begin
        kernel(
            y,
            x,
            w;
            threads,
            blocks,
            shmem = sizeof(Float32) * size(w, 1) * size(w, 2) * size(w, 3),
        )
    end
    return y
end

# flip weights to match conv implementation
w2 = w_g[end:-1:1, end:-1:1, :, :]
y = myconv_2(x_g, w2);
CUDA.@time y = myconv_2(x_g, w2);
@btime myconv_2($x_g, $w2);
