using Revise
using AutoGrad
using BenchmarkTools
using Random: seed!

seed!(123)
bs = 4096
f = 256
h1 = 512
w1 = randn(h1, f) .* 0.01;
b1 = randn(h1);
x = randn(f, bs) .* 0.01;

w1 = Param(w1)
b1 = Param(b1)

autograd_loss(x, w, b) = sum(exp.(w * x .+ b))
loss = @diff autograd_loss(x, w1, b1)
value(loss)
grad(loss, w1)
grad(loss, b1)

# grads = grad(autograd_loss)(x, w1, b1)
# grads, loss = gradloss(autograd_loss)(x, w1, b1)
# grad(grads, w1)

function autograd_test(x, w, b)
    loss = @diff autograd_loss(x, w, b)
    # return nothing
    return value(loss), (w = grad(loss, w), b = grad(loss, b))
end

loss, grads = autograd_test(x, w1, b1)

#  37.210 ms (154 allocations: 81.01 MiB)
@btime autograd_test($x, $w1, $b1);