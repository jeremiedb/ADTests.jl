using Flux
using Flux: onehotbatch, onecold, params, gradient
using MLDatasets: MNIST
using Base.Iterators: partition
using Statistics: mean
using Random: shuffle

#---------------------------------- DATA -------------------------------------
DATA_TRAIN = MNIST(split=:train)[:]
DATA_TEST = MNIST(split=:test)[:]

#-------------------------------- PREPROCESS DATA ------------------------------
x_train = [x for x in eachslice(DATA_TRAIN[1], dims=2)] # reshape to vector of size 28 with matrix of size 28 x 60000
x_test = [x for x in eachslice(DATA_TEST[1], dims=2)] # reshape to vector of size 28 with matrix of size 28 x 10000

# create onehotbatch for train label
y_train = onehotbatch(DATA_TRAIN[2], 0:9)
y_test = DATA_TEST[2]

#------------------------------ CONSTANTS ---------------------------------------
INPUT_DIM = size(x_train[1], 1)
OUTPUT_DIM = 10 # number of classes
LR = 0.001f0 # learning rate
EPOCHS = 10
BATCH_SIZE = 64
TOTAL_SAMPLES = size(x_train[1], 2)

#--------------------------------- BUILD MODEL -----------------------------------
model = Chain(
  RNN(INPUT_DIM => 128, relu),
  Dense(128, OUTPUT_DIM)
)

#----------------------------- HELPER FUNCTIONS --------------------------------------
function loss_fn_2(m, x, y)
  out = [m(xi) for xi in x] # generate output for each of the 28 timesteps
  Flux.Losses.logitcrossentropy(out[end], y) # compute loss based on predictions of the latest timestep
end

function accuracy_eval(m, x, y)
  Flux.reset!(m)
  out = [m(xi) for xi in x]
  mean(onecold(out[end], 0:9) .== y)
end  

θ = params(model) # model parameters to be updated during training
opt = Flux.ADAM(LR) # optimizer function

#---------------------------- RUN TRAINING ----------------------------------------------
for epoch ∈ 1:2
  for idx ∈ partition(1:TOTAL_SAMPLES, BATCH_SIZE)
    features = [x[:, idx] for x ∈ x_train]
    labels = y_train[:, idx]
    Flux.reset!(model)
    gs = gradient(θ) do
      loss = loss_fn_2(model, features, labels)
    end
    # update model
    Flux.Optimise.update!(opt, θ, gs)
  end

  # evaluate model
  @info epoch
#   @show accuracy_eval(model, x_test, y_test)
end


model = model |> gpu
# x_train = x_train |> gpu
# y_train = y_train |> gpu
x_test = x_test |> gpu
y_test = y_test |> gpu

for epoch ∈ 1:2
    for idx ∈ partition(1:TOTAL_SAMPLES, BATCH_SIZE)
      features = [x[:, idx] for x ∈ x_test]
      labels = y_train[:, idx]
      Flux.reset!(model)
      gs = gradient(θ) do
        loss = loss_fn_2(model, features, labels)
      end
      # update model
      Flux.Optimise.update!(opt, θ, gs)
    end
  
    # evaluate model
    @info epoch
  #   @show accuracy_eval(model, x_test, y_test)
  end


model = model |> gpu

loss_fn1(m, X, y) = logitcrossentropy([m(x) for x ∈ X][end], y)
accuracy1(m, X, y) = mean(onecold([m(x) for x ∈ X][end], 0:9) .== y)

rule = Flux.Optimisers.Adam()
opts = Flux.Optimisers.setup(rule, model);

for epoch ∈ 1:2
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