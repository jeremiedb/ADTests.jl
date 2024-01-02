using Revise
# using Statistics
# using Random: seed!
# using Tullio
# using CUDA
# using NNlib
# using Random: seed!
# using NNlibCUDA
using LoopVectorization
using Base.Threads: @threads
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

function my_conv_1(x, w)
    y = zero(x)
    for b in axes(x, 2)
        for hi in axes(x, 1)
            y[hi, b] += w[hi] * x[hi, b]
        end
    end
    return y
end
x = rand(Float32, 3, 8);
w = rand(Float32, 3);
y = my_conv_1(x, w);
@btime y = my_conv_1($x, $w);
# @code_warntype my_conv_1(x, w);

loss1(x, w) = sum(my_conv_1(x, w))
dw = zero(w);
loss1(x, w)
grads = Enzyme.autodiff(Reverse, loss1, Const(x), Duplicated(w, dw));


function my_conv_2(x, w)
    y = zero(x)
    @turbo for b in axes(x, 2)
        for hi in axes(x, 1)
            y[hi, b] += w[hi] * x[hi, b]
        end
    end
    return y
end
# x = rand(Float32, 3, 8);
# w = rand(Float32, 3);
y = my_conv_2(x, w);
@btime y = my_conv_2($x, $w);
# @code_warntype my_conv_2(x, w);

loss2(x, w) = sum(my_conv_2(x, w))
dw = zero(w);
loss2(x, w)
grads = Enzyme.autodiff(Reverse, loss2, Const(x), Duplicated(w, dw));
