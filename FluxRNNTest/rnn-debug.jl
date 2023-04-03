#################################################
# With explicit gradients
#################################################
using Flux
layer1 = Flux.Recur(Flux.RNNCell(1, 1, identity))
layer1.cell.Wi .= 5.0f0
layer1.cell.Wh .= 4.0f0
layer1.cell.b .= 0.0f0
layer1.cell.state0 .= 7.0f0
x = [[2.0f0], [3.0f0]]
Flux.reset!(layer1)
e1, g1 = Flux.withgradient(layer1) do m
    out = [m(xi) for xi in x]
    sum(out[2])
end
# (cell = (σ = nothing, Wi = Float32[14.0;;], Wh = Float32[104.0;;], b = Float32[6.0], state0 = nothing), state = Float32[20.0;;])
gg = g1[1]
gg[:cell]

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
layer2 = Flux.Recur(Flux.RNNCell(1, 1, identity))
layer2.cell.Wi .= 5.0
layer2.cell.Wh .= 4.0
layer2.cell.b .= 0.0f0
layer2.cell.state0 .= 7.0
x = [[2.0f0], [3.0f0]]
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




#################################################
# Full loop
#################################################
using Flux
cell = Flux.RNNCell(1, 1, identity)
cell.Wi .= 5.0f0
cell.Wh .= 4.0f0
cell.b .= 0.0f0
cell.state0 .= 7.0f0
x = [[2.0f0], [3.0f0]]

state = cell.state0 .* 1f0
h, out = cell(state, x[1])
e1, g1 = Flux.withgradient(cell) do m
    state = m.state0
    state, out = m(state, x[1])
    state, out = m(state, x[2])
    sum(out)
end

# function unroll_recur(cell, x)
#     state = cell.state0
#     out = state
#     for i in eachindex(x)
#         state, out = cell(state, x[i])
#     end
#     return out
# end
# unroll_recur(cell, x)
e1, g1 = Flux.withgradient(cell) do m
    state = m.state0
    out = state
    for i in eachindex(x)
        state, out = m(state, x[i])
    end
    sum(out)
end

# (cell = (σ = nothing, Wi = Float32[14.0;;], Wh = Float32[104.0;;], b = Float32[6.0], state0 = nothing), state = Float32[20.0;;])
gg = g1[1]
g1[1]
Flux.reset!(layer1)
out = [layer1(xi) for xi in x]
out1 = layer1.cell.state0 .* layer1.cell.Wh .+ x[1] .* layer1.cell.Wi
out2 = out1 .* layer1.cell.Wh .+ x[2] .* layer1.cell.Wi

∇Wi = x[1] .* layer1.cell.Wh .+ x[2]
∇Wh = 2 .* layer1.cell.Wh .* layer1.cell.state0 .+ x[1] .* layer1.cell.Wi
∇b = layer1.cell.Wh .+ 1
∇state0 = layer1.cell.Wh .^ 2



function test_explicit()
    layer = Flux.Recur(Flux.RNNCell(1, 1, identity))
    layer.cell.Wi .= 5.0f0
    layer.cell.Wh .= 4.0f0
    layer.cell.b .= 0.0f0
    layer.cell.state0 .= 7.0f0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
        layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
        x[2] .* layer.cell.Wi
    ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
    ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
    ∇b = layer.cell.Wh .+ 1
    ∇state0 = layer.cell.Wh .^ 2

    Flux.reset!(layer)
    e, g = Flux.withgradient(layer) do m
        # Flux.reset!(m.cell)
        out = [m(xi) for xi in x]
        sum(out[2])
    end
    grads = g[1][:cell]

    @info grads

    @assert primal[1] ≈ e
    @assert ∇Wi ≈ grads[:Wi]
    @assert ∇Wh ≈ grads[:Wh]
    @assert ∇b ≈ grads[:b]
    @assert ∇state0 ≈ grads[:state0]
    return nothing
end
test_explicit()


function test_loop()
    cell = Flux.RNNCell(1, 1, identity)
    cell.Wi .= 5.0f0
    cell.Wh .= 4.0f0
    cell.b .= 0.0f0
    cell.state0 .= 7.0f0
    x = [[2.0f0], [3.0f0]]

    # theoretical primal gradients
    primal =
        cell.Wh .* (cell.Wh * cell.state0 .+ x[1] .* cell.Wi) .+
        x[2] .* cell.Wi
    ∇Wi = x[1] .* cell.Wh .+ x[2]
    ∇Wh = 2 .* cell.Wh .* cell.state0 .+ x[1] .* cell.Wi
    ∇b = cell.Wh .+ 1
    ∇state0 = cell.Wh .^ 2

    Flux.reset!(cell)
    e, g = Flux.withgradient(cell) do m
        # Flux.reset!(m)
        state = m.state0
        out = state
        for i in eachindex(x)
            state, out = m(state, x[i])
        end
        sum(out)
    end
    grads = g[1]

    @info grads
    @assert primal[1] ≈ e
    @assert ∇Wi ≈ grads[:Wi]
    @assert ∇Wh ≈ grads[:Wh]
    @assert ∇b ≈ grads[:b]
    @assert ∇state0 ≈ grads[:state0]
    return nothing
end
test_loop()