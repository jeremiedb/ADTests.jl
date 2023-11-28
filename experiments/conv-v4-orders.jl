using Revise
using Statistics
using Random: seed!
using LoopVectorization
# using Tullio
# using CUDA
using Base.Threads: @threads
using NNlib
using Random: seed!
# using NNlibCUDA
using Enzyme
using BenchmarkTools

####################################################################################################################################
# Skip LV to see if Enzyme can handle it
####################################################################################################################################

# width, height, channels, batchsize
CI = 8
CO = 16
B = 32
W = 28
H = 28
KW = KH = 3

function _dot_prod_A!(C, A, B)
    for n ∈ axes(A, 1)
        Cn = zero(eltype(C))
        for k ∈ eachindex(B)
            Cn += A[n, k] * B[k]
        end
        C[n] = Cn
    end
    return nothing
end
function _dot_prod_A_lv!(C, A, B)
    @turbo for n ∈ axes(A, 1)
        Cn = zero(eltype(C))
        for k ∈ eachindex(B)
            Cn += A[n, k] * B[k]
        end
        C[n] = Cn
    end
    return nothing
end
function _dot_prod_B_lv!(C, A, B)
    @turbo for n ∈ axes(B, 2)
        Cn = zero(eltype(C))
        for k ∈ eachindex(A)
            Cn += A[k] * B[k, n]
        end
        C[n] = Cn
    end
    return nothing
end

function _point_conv_A!(y, x, w)
    _dot_prod_A_lv!(y, x, w)
    return y
end
function _point_conv_B!(y, x, w)
    _dot_prod_B_lv!(y, x, w)
    return y
end

w = randn(Float32, CO, CI, KW, KH);
x = randn(Float32, CI, W, H, B);
y = zeros(Float32, CO, W - 2, H - 2, B);

xi = x[:, 1:3, 1:3, 1]
# xi = view(x, :, 1:3, 1:3, 1)
xi_flat = reshape(xi, :)
w_flat = reshape(w, CO, :)

yi = zeros(Float32, CO)
_dot_prod_A!(yi, w_flat, xi_flat)
# index: 497.082 ns (0 allocations: 0 bytes)
# view: 3.853 μs (0 allocations: 0 bytes)
@btime _dot_prod_A!($yi, $w_flat, $xi_flat)


xi = x[:, 1:3, 1:3, 1]
# xi = view(x, :, 1:3, 1:3, 1)
xi_flat = reshape(xi, :)
w_flat = reshape(w, CO, :)

yi = zeros(Float32, CO)
_dot_prod_A_lv!(yi, w_flat, xi_flat)
# index: 71.873 ns (0 allocations: 0 bytes)
# view: error - input not strided
@btime _dot_prod_A_lv!($yi, $w_flat, $xi_flat)
# 55.937 ns (0 allocations: 0 bytes)
@btime _dot_prod_B_lv!($yi, $xi_flat, $w_flat')


######################
# switch x CI order
x = randn(Float32, W, H, CI, B);
xi = x[1:3, 1:3, :, 1]
# xi = view(x, 1:3, 1:3, :, 1)
xi_flat = reshape(xi, :)
w_flat = reshape(w, CO, :)

yi = zeros(Float32, CO)
_dot_prod_A_lv!(yi, w_flat, xi_flat)
# index: 71.873 ns (0 allocations: 0 bytes)
# view: error - input not strided
@btime _dot_prod_A_lv!($yi, $w_flat, $xi_flat)
# 54.520 ns (0 allocations: 0 bytes)
@btime _dot_prod_B_lv!($yi, $xi_flat, $w_flat')

######################
# switch w order
w = randn(Float32, KW, KH, CI, CO);
x = randn(Float32, W, H, CI, B);
xi = x[1:3, 1:3, :, 1]
# xi = view(x, 1:3, 1:3, :, 1)
xi_flat = reshape(xi, :)
w_flat = reshape(w, :, CO)

yi = zeros(Float32, CO)
_dot_prod_A_lv!(yi, w_flat', xi_flat)
# 37.632 ns (0 allocations: 0 bytes)
@btime _dot_prod_B_lv!($yi, $xi_flat, $w_flat)
# index: 29.237 ns (0 allocations: 0 bytes)
@btime _dot_prod_A_lv!($yi, $w_flat', $xi_flat)


function my_conv_4!(y, x, w)
    y .= 0
    szw = size(w)
    w_flat = reshape(w, :, last(szw))
    @threads for b in axes(x, 4)
        yi = zeros(Float32, last(szw))
        for hi in axes(y, 2)
            for wi in axes(y, 1)
                xi_flat = reshape(x[wi:wi+2, hi:hi+2, :, b], :)
                # view(y, wi, hi, :, b) .= _point_conv_A!(yi, w_flat', xi_flat)
                view(y, wi, hi, :, b) .= _point_conv_B!(yi, xi_flat, w_flat)
            end
        end
    end
    return nothing
end

function my_conv_4(x::T, w::T) where {T}
    szx = size(x)
    szw = size(w)
    y = zeros(Float32, szx[1] - 2, szx[2] - 2, szw[4], szx[4])::Array{Float32, 4}
    my_conv_4!(y, x, w)
    return y
end

w = randn(Float32, KW, KH, CI, CO);
x = randn(Float32, W, H, CI, B);

@time y = my_conv_4(x, w);
# A: 971.197 μs (64994 allocations: 10.57 MiB)
# B: 1.156 ms (64994 allocations: 10.57 MiB)
@btime my_conv_4($x, $w);
std(y)
@code_warntype my_conv_4(x, w);
