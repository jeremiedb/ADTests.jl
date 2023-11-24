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
# This approach fetches the full receptive field of the kernel and directly compute the output channel of a given location (W,H)
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
w = randn(Float32, CO, CI, KW, KH);
x = randn(Float32, CI, W, H, B);
y = zeros(Float32, CO, W - 2, H - 2, B);

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


function _dot_prod_lv2B!(C, A, B)
    @turbo for n ∈ axes(A, 1)
        Cn = zero(eltype(C))
        for k ∈ eachindex(B)
            Cn += A[n, k] * B[k]
        end
        C[n] = Cn
    end
end

function _point_conv_lv2B!(y, x, w)    
    _dot_prod_lv2B!(y, x, w)
    return y
end

xi = x[:, 1:3, 1:3, 1]
# xi = view(x, :, 1:3, 1:3, 1)
xi_flat = reshape(xi, :)
yi = zeros(Float32, CO)
w_flat = reshape(w, CO, :)
# @btime w_flat * xi_flat;
_point_conv_lv2B!(yi, w_flat, xi_flat)
# @btime _point_conv_lv2B!($yi, $w_flat, $xi_flat);
# @time yi_out = _point_conv_lv2B!(yi, xi, w_flat);
# @btime yi_out = _point_conv_lv2B!(yi, xi, w_flat);

function my_conv_2B!(y, x, w)
    y .= 0
    w_flat = reshape(w, CO, :)::Array{Float32,2}
    @threads for b in axes(x, 4)
        yi = zeros(Float32, 16)
        for hi in axes(y, 3)
            for wi in axes(y, 2)
                xi = reshape(x[:, wi:wi+2, hi:hi+2, b], :)
                view(y, :, wi, hi, b) .= _point_conv_lvB!(yi, w_flat, xi)
            end
        end
    end
    return nothing
end

@time my_conv_2B!(y, x, w);
# 834.700 μs (50339 allocations: 2.33 MiB)
@btime my_conv_2B!($y, $x, $w);
std(y)
@code_warntype my_conv_2B!(y, x, w);

function my_conv_3B(x::T, w::T) where {T}
    y = zeros(Float32, CO, W - 2, H - 2, B)::Array{Float32, 4}
    my_conv_3B!(y, x, w)
    return y
end

@time y3 = my_conv_2B(x, w);
# 1.014 ms (50343 allocations: 4.09 MiB)
@btime my_conv_2B($x, $w);
std(y3)

@code_warntype my_conv_2B(x, w);

using Enzyme
loss(x, w) = sum(my_conv_2B(x, w))
dw = zero(w);
@time loss(x, w)
grads = Enzyme.autodiff(Reverse, loss, Const(x), Duplicated(w, dw));
