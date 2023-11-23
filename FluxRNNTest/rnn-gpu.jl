using Flux
using ChainRulesCore
using CUDA
using Flux.Losses: mse
dev = gpu # cpu is working fine

m = Chain(RNN(2 => 3), Dense(3 => 1)) |> dev
x = [rand(Float32, 2, 5) for i = 1:4] |> dev;
y = [rand(Float32, 1, 5) for i = 1:1] |> dev

m.layers[1].state
Flux.reset!(m)
m.layers[1].state
[m(xi) for xi in x]

function loss(m, x, y)
    # @ignore_derivatives Flux.reset!(m)Flux.reset!(m)
    # m(x[1]) # ignores the output but updates the hidden states
    # m(x[2]) # ignore second output
    mse(m(x[3]), y)
end

y = rand(Float32, 1, 5) |> dev
loss(m, x, y)

Flux.reset!(m)
# grads = Flux.gradient(m, x, y) do m, x, y
#     loss(m, x, y)
# end

gs = gradient(m) do model
    loss(model, x, y)
end

function get_grads(m, x, y)
    gs = gradient(m) do model
        loss(model, x, y)
    end
    return gs
end

optim = Flux.setup(Flux.Adam(), m)
Flux.update!(optim, m, grads[1])

function train!(m, x, y)
    Flux.reset!(m)
    grads = Flux.gradient(m) do model
        loss(model, x, y)
    end
    return grads
end

train!(m, x, y)




# using ChainRulesCore
using Flux
using CUDA
using Flux.Losses: mse
dev = gpu # cpu is working fine

#######################
# no indexing
#######################
m = RNN(2 => 1) |> dev
x = rand(Float32, 2, 3) |> dev;
y = rand(Float32, 1, 3) |> dev

m.state
Flux.reset!(m)
m(x)

loss(m, x, y) = mse(m(x), y)
Flux.reset!(m)
gs = gradient(m) do model
    loss(model, x, y)
end

#######################
# with indexing
#######################
m = RNN(2 => 1) |> dev
x = [rand(Float32, 2, 3) for i in 1:4] |> dev;
y = rand(Float32, 1, 3) |> dev

m.state
Flux.reset!(m)
m(x[1])

function loss_1(m, x, y)
    p = [m(xi) for xi in x]
    mse(p[1], y)
end

loss_1(m, x, y)

Flux.reset!(m)
gs = gradient(m) do model
    loss_1(model, x, y)
end

function get_grads(m, x, y)
    gs = gradient(m) do model
        loss_1(model, x, y)
    end
    return gs
end

get_grads(m, x, y)



function loss_2(m, x, y)
    p = m(x[1])
    mse(p, y)
end

loss_2(m, x, y)

Flux.reset!(m)
gs = gradient(m) do model
    loss_2(model, x, y)
end

function get_grads(m, x, y)
    gs = gradient(m) do model
        loss_1(model, x, y)
    end
    return gs
end

get_grads(m, x, y)
