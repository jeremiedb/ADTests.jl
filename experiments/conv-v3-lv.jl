using Revise
# using Random: seed!
# using Base.Threads: @threads
using LoopVectorization
using Enzyme
# using BenchmarkTools

####################################################################################################################################
# Skip LV to see if Enzyme can handle it
####################################################################################################################################
# function my_conv_A(x, w)
#     y = zero(x)
#     for hi in axes(x, 2)
#         for k in axes(x, 1)
#             y[k] += w[k] * x[k, hi]
#         end
#     end
#     return y
# end
# x = rand(Float32, 3, 5);
# w = rand(Float32, 3);
# y = my_conv_A(x, w);
# loss_A(x, w) = sum(my_conv_A(x, w))
# dw = zero(w);
# loss_A(x, w)
# @code_warntype loss_A(x, w)
# grads = Enzyme.autodiff(Reverse, loss_A, Const(x), Duplicated(w, dw));


function my_conv_B(x, w)
    y = zero(x)
    @turbo for hi in axes(x, 2)
        for k in axes(x, 1)
            y[k] += w[k] * x[k, hi]
        end
    end
    return y
end
x = rand(Float32, 3, 5);
w = rand(Float32, 3);
y = my_conv_B(x, w);
loss_B(x, w) = sum(my_conv_B(x, w))
dw = zero(w);
loss_B(x, w)
# @code_warntype loss_B(x, w)
grads = Enzyme.autodiff(Reverse, loss_B, Const(x), Duplicated(w, dw));
