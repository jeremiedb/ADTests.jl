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


function _dot_prod_3!(C, A, B)
    for n ∈ axes(A, 1)
        Cn = zero(eltype(C))
        for k ∈ eachindex(B)
            Cn += A[n, k] * B[k]
        end
        C[n] = Cn
    end
end

function _point_conv_3!(y, x, w)
    _dot_prod_3!(y, x, w)
    return y
end

xi = x[:, 1:3, 1:3, 1]
# xi = view(x, :, 1:3, 1:3, 1)
xi_flat = reshape(xi, :)
yi = zeros(Float32, CO)
w_flat = reshape(w, CO, :)
# @btime w_flat * xi_flat;
_point_conv_3!(yi, w_flat, xi_flat)
# @btime _point_conv_3!($yi, $w_flat, $xi_flat);
# @time yi_out = _point_conv_3!(yi, xi, w_flat);
# @btime yi_out = _point_conv_3!(yi, xi, w_flat);

function my_conv_3A!(y, x, w_flat)
    y .= 0
    # w_flat = reshape(w, CO, 72)::Array{Float32,2}
    for b in axes(y, 4)
        yi = zeros(Float32, 16)
        for hi in axes(y, 3)
            for wi in axes(y, 2)
                xi = reshape(x[:, wi:wi+2, hi:hi+2, b], :)
                view(y, :, wi, hi, b) .= _point_conv_3!(yi, w_flat, xi)
            end
        end
    end
    return nothing
end

w_flat = reshape(w, CO, :)
@time my_conv_3A!(y, x, w_flat);
# 834.700 μs (50339 allocations: 2.33 MiB)
@btime my_conv_3A!($y, $x, $w_flat);
std(y)
@code_warntype my_conv_3A!(y, x, w_flat);

function my_conv_3A(x::T, w::A) where {T,A}
    y = zeros(Float32, 16, 26, 26, 32)::Array{Float32,4}
    my_conv_3A!(y, x, w)
    return y
end

@time y3 = my_conv_3A(x, w_flat);
# 1.014 ms (50343 allocations: 4.09 MiB)
@btime my_conv_3A($x, $w_flat);
std(y3)

@code_warntype my_conv_3A(x, w_flat);

loss(x, w) = sum(my_conv_3A(x, w))
dw = zero(w_flat);
@time loss(x, w_flat)
grads = Enzyme.autodiff(Reverse, loss, Const(x), Duplicated(w_flat, dw));



function my_conv_mini(x, w)
    szx = size(x)
    y = zeros(eltype(x), szx[1], szx[2])
    for b in axes(y, 2)
        y[:, b] .= w .* x[:, b]
    end
    return y
end
x = rand(Float32, 3, 5);
w = rand(Float32, 3);
y = my_conv_mini(x, w);
loss(x, w) = sum(my_conv_mini(x, w))
dw = zero(w);
loss(x, w)
grads = Enzyme.autodiff(Reverse, loss, Const(x), Duplicated(w, dw));



using Enzyme

function my_conv_1(x, w)
    y = zero(x)
    for b in axes(y, 3)
        for wi in axes(y, 2)
            y[:, wi, b] .= w .* x[:, wi, b]
        end
    end
    return y
end
x = rand(Float32, 3, 5, 8);
w = rand(Float32, 3);
y = my_conv_1(x, w);
loss1(x, w) = sum(my_conv_1(x, w))
dw = zero(w);
loss1(x, w)
grads = Enzyme.autodiff(Reverse, loss1, Const(x), Duplicated(w, dw));


function my_conv_2(x, w)
    y = zero(x)
    for b in axes(y, 4)
        for hi in axes(y, 3)
            for wi in axes(y, 2)
                y[:, wi, hi, b] .= w .* x[:, wi, hi, b]
            end
        end
    end
    return y
end
x = rand(Float32, 3, 5, 5, 8);
w = rand(Float32, 3);
y = my_conv_2(x, w);
loss2(x, w) = sum(my_conv_2(x, w))
dw = zero(w);
loss2(x, w)
grads = Enzyme.autodiff(Reverse, loss2, Const(x), Duplicated(w, dw));


function my_conv_3(x, w)
    y = zero(x)
    for hi in axes(y, 3)
        y[1] += w[1] * x[1]
    end
    return y
end
x = rand(Float32, 3, 5, 5, 8);
w = rand(Float32, 3);
y = my_conv_3(x, w);
loss3(x, w) = sum(my_conv_3(x, w))
dw = zero(w);
loss3(x, w)
grads = Enzyme.autodiff(Reverse, loss3, Const(x), Duplicated(w, dw));
