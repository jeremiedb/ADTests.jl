# using Revise
# using Random: seed!
using Enzyme
using NNlib

loss(w, x) = sum(conv(x, w))
#############################################
# im2col algo
#############################################
w = randn(Float32, 3, 3, 5, 7);
dw = zero(w);
# m(w, x) = conv(x, w)
x = randn(Float32, (3, 3, 5, 8));
# size(x)

# @time m(w, x);
@time loss(w, x);
grads = Enzyme.autodiff(Reverse, loss, Duplicated(w, dw), Const(x));


@code_warntype conv(x, w);
@code_lowered conv(x, w);
@code_typed conv(x, w);
@code_llvm conv(x, w);
@code_native conv(x, w);


#############################################
# direct conv algo
#############################################
w = randn(Float32, 3, 3, 5, 7);
dw = zero(w);
x = randn(Float64, (3, 3, 5, 8));
dx = zero(x);

# @time m(w, x);
@time loss(w, x);
grads = Enzyme.autodiff(Reverse, loss, Duplicated(w, dw), Const(x));
grads = Enzyme.autodiff(Reverse, loss, Duplicated(w, dw), Duplicated(x, dx));


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
