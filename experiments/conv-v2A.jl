using Revise
using Statistics
using Random: seed!
using LoopVectorization
using Tullio
# using CUDA
using Base.Threads: @threads
using NNlib
using Random: seed!
# using NNlibCUDA
using BenchmarkTools

####################################################################################################################################
# This approach takes a single x location (W,H) and compute the effect over the full output kernel
# The complete effect on y location (W,H) is the sum of all the iterations of x locations that overlay with it
####################################################################################################################################

# width, height, channels, batchsize
CI = 8
CO = 16
B = 32
W = 28
H = 28
KW = KH = 3

# C,W,H,B
# seed!(123)
w = randn(Float32, CI, CO, KW, KH);
x = randn(Float32, CI, W, H, B);
y = zeros(Float32, CO, W + 2, H + 2, B);

########################
# NNlib ref
########################
x_nlib = randn(Float32, W, H, CI, B);
w_nlib = randn(Float32, KW, KH, CI, CO)
@time y_nlib = NNlib.conv(x_nlib, w_nlib);
# 2.036 ms (128 allocations: 4.31 MiB)
@btime y_nlib = NNlib.conv($x_nlib, $w_nlib);
size(y_nlib)
std(y_nlib)


@inline _point_conv(x, w) = dropdims(sum(x .* w; dims=1); dims=1)
@inline _point_conv!(y, x, w) = dropdims(sum!(y, x .* w); dims=1)

xi = x[:, 1, 1, 1]
yi = zeros(Float32, 1, FO, KW, KH)

@time pci = _point_conv(xi, w);
@time _point_conv!(yi, xi, w);
# @btime _point_conv($xi, $w);
# @btime _point_conv!($yi, $xi, $w);

function my_conv_1!(y, x, w)
    y .= 0
    @threads for b in axes(x, 4)
        @inbounds for hi in axes(x, 3)
            for wi in axes(x, 2)
                xi = view(x, :, wi, hi, b)
                view(y, :, wi:wi+2, hi:hi+2, b) .+= _point_conv(xi, w)
            end
        end
    end
    return nothing
end

# y .= 0;
@time my_conv_1!(y, x, w)
# 21.966 ms (301153 allocations: 137.06 MiB)
@btime my_conv_1!($y, $x, $w)
std(y)
std(y[:, 3:end-2, 3:end-2, :])

function my_conv_2!(y, x, w)
    y .= 0
    @threads for b in axes(x, 4)
        yi = zeros(Float32, 1, 16, 3, 3)
        @inbounds for hi in axes(x, 3)
            @inbounds for wi in axes(x, 2)
                xi = view(x, :, wi, hi, b)
                view(y, :, wi:wi+2, hi:hi+2, b) .+= _point_conv!(yi, xi, w)
            end
        end
    end
    return nothing
end

@time my_conv_2!(y, x, w)
# 17.489 ms (251041 allocations: 119.85 MiB)
@btime my_conv_2!($y, $x, $w)
std(y)
std(y[:, 3:end-2, 3:end-2, :])


function _dot_prod_lv!(C, A, B)
    @turbo for n ∈ axes(B, 2)
        Cn = zero(eltype(C))
        for k ∈ eachindex(A)
            Cn += A[k] * B[k, n]
        end
        C[n] = Cn
    end
end

function _point_conv_lv!(y, x, w)    
    _dot_prod_lv!(y, x, w)
    return reshape(y, 16, 3, 3)
end

xi = x[:, 1, 1, 1]
yi = zeros(Float32, CO * KW * KH)
w_flat = reshape(w, 8, :)
# _dot_prod_lv!(yi, xi, w_flat)
# @btime _dot_prod_lv!(yi, xi, w_flat)
# @time yi_out = _point_conv_lv!(yi, xi, w_flat);
# @btime yi_out = _point_conv_lv!(yi, xi, w_flat);

function my_conv_3!(y, x, w)
    y .= 0
    w_flat = reshape(w, 8, :)
    @threads for b in axes(x, 4)
        yi = zeros(Float32, 1, 16, 3, 3)
        for hi in axes(x, 3)
            for wi in axes(x, 2)
                xi = reshape(view(x, :, wi, hi, b), 8)
                view(y, :, wi:wi+2, hi:hi+2, b) .+= _point_conv_lv!(yi, xi, w_flat);
            end
        end
    end
    return nothing
end

@time my_conv_3!(y, x, w);
# 834.700 μs (50339 allocations: 2.33 MiB)
@btime my_conv_3!($y, $x, $w);
std(y)
std(y[:, 3:end-2, 3:end-2, :])

@code_warntype my_conv_3!(y, x, w);

function my_conv_3(x::T, w::T) where {T}
    y = zeros(Float32, CO, W + 2, H + 2, B)::Array{Float32, 4}
    my_conv_3!(y, x, w)
    return y
end

@time y3 = my_conv_3(x, w);
# 1.014 ms (50343 allocations: 4.09 MiB)
@btime my_conv_3($x, $w);
std(y3)
std(y3[:, 3:end-2, 3:end-2, :])

@code_warntype my_conv_3(x, w);

using Enzyme
loss(x, w) = sum(my_conv_3(x, w))
dw = zero(w);
@time loss(x, w)
grads = Enzyme.autodiff(Reverse, loss, Const(x), Duplicated(w, dw));
