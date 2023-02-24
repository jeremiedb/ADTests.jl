# using Revise
# using Random: seed!
using Enzyme
using NNlib
# using Flux

#############################################
# direct params
#############################################
w = randn(Float32, 3, 3, 5, 7);
dw = zero(w)
# m(w, x) = conv(x, w)
loss(w, x) = sum(conv(x, w))
x = randn(Float32, (3, 3, 5, 8));
# size(x)

@time m(w, x);
@time loss(w, x);
grads = Enzyme.autodiff(loss, Duplicated(w, dw), Const(x));

#############################################
# Custom struct
#############################################
# struct MyConv{A}
#     w::A
# end
# (m::MyConv)(x) = conv(x, m.w)

# m = MyConv(randn(3, 3, 5, 7));
# dm = Flux.fmap(m) do x
#     x isa Array ? zero(x) : x
# end

# loss(m, x) = sum(m(x))
# x = randn(Float32, (3, 3, 5, 8));
# size(x)

# @time m(x);
# @time loss(m, x);
# grads = Enzyme.autodiff(loss, Duplicated(m, dm), Const(x));

#############################################
# Flux
#############################################
# model = Chain(Conv((3, 3), 3 => 5))
# dmodel = Flux.fmap(model) do x
#     x isa Array ? zero(x) : x
# end

# loss(model, x) = sum(model(x))

# x = randn(Float32, (3, 3, 3, 8));
# size(x)

# @time model(x);
# @time loss(model, x);
# grads = Enzyme.autodiff(loss, Duplicated(model, dmodel), Const(x));
