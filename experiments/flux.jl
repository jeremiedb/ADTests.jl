using Flux
using Flux: gradient, logitcrossentropy, params, Momentum
using OneHotArrays: onecold, onehotbatch
using MLDatasets: MNIST
using Random: shuffle
using Statistics: mean
using Base.Iterators: partition

# ------------------- data --------------------------
train_x, train_y = MNIST(split=:train).features, MNIST(split=:train).targets
test_x, test_y = MNIST(split=:test).features, MNIST(split=:test).targets
train_y = onehotbatch(train_y, 0:9)
train_x = [x for x ∈ eachslice(train_x, dims=2)]
test_x = [x for x ∈ eachslice(test_x, dims=2)]
# ------------------ constants ---------------------
INPUT_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 5
# ------------------ model --------------------------
model = Chain(
    RNN(INPUT_SIZE, 128, relu),
    RNN(128, 64, relu),
    Dense(64, NUM_CLASSES)
)
# ---------------- helper --------------------------
loss_fn1(m, X, y) = logitcrossentropy([m(x) for x ∈ X][end], y)
accuracy1(m, X, y) = mean(onecold([m(x) for x ∈ X][end], 0:9) .== y)

loss_fn2(X, y) = logitcrossentropy([model(x) for x ∈ X][end], y)
accuracy2(X, y) = mean(onecold([model(x) for x ∈ X][end], 0:9) .== y)

function loss_fn3(X, y)
    Flux.Zygote.@ignore Flux.reset!(model)
    logitcrossentropy([model(x) for x ∈ X][end], y)
end

function accuracy3(X, y)
    Flux.Zygote.@ignore Flux.reset!(model)
    @info size(model.layers[1].state)
    mean(onecold([model(x) for x ∈ X][end], 0:9) .== y)
end

opt = Flux.Optimise.Adam()
ps = params(model)

# --------------- train -----------------------------
for epoch ∈ 1:5
    for idx ∈  partition(shuffle(1:size(train_y, 2)), BATCH_SIZE)
        X = [x[:, idx] for x ∈ train_x]
        y = train_y[:, idx]
        # @info "size X" size(X)
        # @info "size y" size(y)
        ps = params(model)
        Flux.reset!(model)
        gs = gradient(ps) do 
            loss_fn1(model, X, y)
            # loss_fn2(X, y)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    Flux.reset!(model)
    test_acc = accuracy1(model, test_x, test_y)
    # test_acc = accuracy2(test_x, test_y)
    @info "Epoch : $epoch | accuracy : $test_acc"
end



#############################
# explicit
#############################
rule = Flux.Optimisers.Adam(1e-3)  # use the Adam optimiser with its default settings
opts = Flux.Optimisers.setup(rule, model);  # initialise this optimiser's momentum etc.

# --------------- train -----------------------------
for epoch ∈ 1:5
    for idx ∈  partition(shuffle(1:size(train_y, 2)), BATCH_SIZE)
        X = [x[:, idx] for x ∈ train_x]
        y = train_y[:, idx]
        Flux.reset!(model)
        gs = gradient(model) do m
            loss_fn1(m, X, y)
        end
        Flux.Optimisers.update!(opts, model, gs[1]);
    end
    Flux.reset!(model)
    test_acc = accuracy1(model, test_x, test_y)
    @info "Epoch : $epoch | accuracy : $test_acc"
end




using Flux 
using Random
Random.seed!(149)

layer1 = Flux.Recur(Flux.RNNCell(1 => 1, identity))

x = Float32[0.8, 0.9]
y = Float32(-0.7)

layer1([x[1]])

Flux.reset!(layer1)
e1, g1 = Flux.withgradient(layer1) do m
    yhat = 0.0
    for i in 1:2 
        yhat = m([x[i]])
    end
    loss = Flux.mse(yhat, y)
    # println(loss)
    return loss 
end
println("flux gradients: ", g1[1])

Flux.reset!(layer1)
e2, g2 = Flux.withgradient(layer1) do m
    yhat = [m([x[i]]) for i in 1:2]
    loss = Flux.mse(yhat[end], y)
    # println(loss)
    return loss 
end
println("flux gradients: ", g2[1])


Flux.reset!(layer1)
layer1([x[1]])
e3, g3 = Flux.withgradient(layer1) do m
    yhat = m([x[2]])
    loss = Flux.mse(yhat, y)
    # println(loss)
    return loss 
end
println("flux gradients: ", g3[1])





#################################################
# Inconsistent state
using Flux 
using Random
Random.seed!(149)
layer1 = Flux.Recur(Flux.RNNCell(3 => 2, identity))
x = rand(Float32, 3)
y = Float32(-0.7)
Flux.reset!(layer1)
e1, g1 = Flux.withgradient(layer1) do m
    sum(m(x))
end
g1[1]
layer1.cell.state0

using Flux 
using Random
Random.seed!(149)
layer1 = Flux.Recur(Flux.RNNCell(3 => 1, identity))
x = rand(Float32, 3)
y = Float32(-0.7)
ps = Flux.params(layer1)
Flux.reset!(layer1)
e1, g1 = Flux.withgradient(ps) do
    sum(layer1(x))
end

g1[ps[4]]
layer1.cell.state0




#################################################
# With explicit gradients
#################################################
using Flux 
layer1 = Flux.Recur(Flux.RNNCell(1 => 1, identity))
layer1.cell.Wi .= 5f0
layer1.cell.Wh .= 4f0
layer1.cell.b .= 0f0
layer1.cell.state0 .= 7f0
x = [[2f0], [3f0]]
Flux.reset!(layer1)
e1, g1 = Flux.withgradient(layer1) do m
    out = [m(xi) for xi in x]
    sum(out[2])
end
g1[1]
# (cell = (σ = nothing, Wi = Float32[14.0;;], Wh = Float32[104.0;;], b = Float32[6.0], state0 = nothing), state = Float32[20.0;;])

Flux.reset!(layer1)
out = [layer1(xi) for xi in x]
out1 = layer1.cell.state0 .* layer1.cell.Wh .+ x[1] .* layer1.cell.Wi
out2 = out1 .* layer1.cell.Wh .+ x[2] .* layer1.cell.Wi

∇Wi = x[1] .* layer1.cell.Wh .+ x[2] 
∇Wh = 2 .* layer1.cell.Wh .* layer1.cell.state0 .+ x[1] .* layer1.cell.Wi 
∇b = layer1.cell.Wh .+ 1
∇state0 = layer1.cell.Wh .^ 2

###########################
# implicit gradients
###########################
using Flux 
layer2 = Flux.Recur(Flux.RNNCell(1 => 1, identity))
layer2.cell.Wi .= 5.0
layer2.cell.Wh .= 4.0
layer2.cell.b .= 0f0
layer2.cell.state0 .= 7.0
x = [[2f0], [3f0]]
Flux.reset!(layer2)
ps = Flux.params(layer2)
e2, g2 = Flux.withgradient(ps) do
    out = [layer2(xi) for xi in x]
    sum(out[2])
end
g2.grads
g2[ps[1]]
g2[ps[2]]
g2[ps[3]]
g2[ps[4]]
layer2.cell.state0
# IdDict{Any, Any} with 7 entries:
#   Float32[0.0]           => Fill(1.0, 1)
#   Float32[4.0;;]         => Float32[38.0;;]
#   Float32[5.0;;]         => Float32[3.0;;]
#   Float32[7.0;;]         => nothing
#   :(Main.x)              => Union{Nothing, Vector{Float32}}[nothing, Float32[5.0]]
#   :(Main.layer2)         => RefValue{Any}((cell = nothing, state = Float32[4.0;;]))
#   Recur(RNNCell(1 => 1)) => RefValue{Any}((cell = nothing, state = Float32[4.0;;]))


##################
# gradient of all outputs
##################
using Flux 
layer1 = Flux.Recur(Flux.RNNCell(1 => 1, identity))
layer1.cell.Wh .= 4.0
layer1.cell.Wi .= 5.0
layer1.cell.state0 .= 7.0
x = [[2f0], [3f0]]
Flux.reset!(layer1)
ps = Flux.params(layer1)
e3, g3 = Flux.withgradient(ps) do
    out = [layer1(xi) for xi in x]
    sum(sum(out))
end
g3.grads
g3[ps[1]]
g3[ps[2]]
g3[ps[3]]
g3[ps[4]]
layer1.cell.state0
