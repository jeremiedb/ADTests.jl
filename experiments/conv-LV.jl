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
function my_conv_1(x, w)
    y = zero(x)
    for b in axes(x, 2)
        for hi in axes(x, 1)
            y[hi, b] += w[hi] * x[hi, b]
        end
    end
    return y
end
x = rand(Float32, 3, 5);
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
