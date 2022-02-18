##################################
# Source code of Neural Net (NN) #
##################################

using Random


struct NeuralNet
  L::Int64                        # number of layers
  n::Vector{Int64}                # sizes of layers
  h::Vector{Vector{Float64}}      # units field
  ξ::Vector{Vector{Float64}}      # units activation
  
  w::Vector{Array{Float64,2}}     # weights
  θ::Vector{Vector{Float64}}      # thresholds

  Δ::Vector{Vector{Float64}}      # deltas
  
  δw::Vector{Array{Float64,2}}    # changes of the weights
  δθ::Vector{Vector{Float64}}     # changes of the thresholds

  δwᵖʳᵉᵛ::Vector{Array{Float64,2}}  # previous changes of the weights
  δθᵖʳᵉᵛ::Vector{Vector{Float64}}   # previous changes of the thresholds
end

function NeuralNet(layers::Vector{Int64})
  L = length(layers)
  n = copy(layers)

  h = Vector{Float64}[]
  ξ = Vector{Float64}[]
  θ = Vector{Float64}[]
  Δ = Vector{Float64}[]
  δθ = Vector{Float64}[]
  δθᵖʳᵉᵛ = Vector{Float64}[]
  for ℓ in 1:L
    push!(h, zeros(layers[ℓ]))
    push!(ξ, zeros(layers[ℓ]))
    push!(θ, rand(layers[ℓ]) .* rand((-1, 1), layers[ℓ]))   # random positive and negative values
    push!(Δ, zeros(layers[ℓ]))
    push!(δθ, zeros(layers[ℓ]))
    push!(δθᵖʳᵉᵛ, zeros(layers[ℓ]))
  end

  w = Array{Float64,2}[]
  δw = Array{Float64,2}[]
  δwᵖʳᵉᵛ = Array{Float64,2}[]
  push!(w, zeros(1, 1))                          # unused, but needed to ensure w[2] refers to weights between the first two layers
  push!(δw, zeros(1, 1))
  push!(δwᵖʳᵉᵛ, zeros(1, 1))
  for ℓ in 2:L
    push!(w, rand(layers[ℓ], layers[ℓ - 1]) .* rand((-1, 1), layers[ℓ], layers[ℓ - 1]))     # random positive and negative values
    push!(δw, zeros(layers[ℓ], layers[ℓ - 1]))
    push!(δwᵖʳᵉᵛ, zeros(layers[ℓ], layers[ℓ - 1]))
  end

  return NeuralNet(L, n, h, ξ, w, θ, Δ, δw, δθ, δwᵖʳᵉᵛ, δθᵖʳᵉᵛ)
end


function sigmoid(h::Float64)::Float64
  # activation function g, Eq. (10)
  return 1 / (1 + exp(-h))
end


function relu(h::Float64)::Float64
  # activation function g
  return max(0, h)
end


function sigmoid_derivative(h::Float64)::Float64
  # g': derivative of activation function g, Eq. (13)
  return sigmoid(h) * (1 - sigmoid(h))
end


function relu_derivative(h::Float64)::Float64 
  # g': derivative of activation function g
  if h >= 0 return 1
  else return 0
  end
end


function feed_forward!(nn::NeuralNet, x_in::Vector{Float64}, y_out::Vector{Float64})
  # copy input to first layer, Eq. (6)
  nn.ξ[1] .= x_in

  # feed-forward of input pattern
  for ℓ in 2:nn.L
    for i in 1:nn.n[ℓ]
      # calculate input field to unit i in layer ℓ, Eq. (8)
      h = -nn.θ[ℓ][i]
      for j in 1:nn.n[ℓ - 1]
        h += nn.w[ℓ][i, j] * nn.ξ[ℓ - 1][j]
      end
      # save field and calculate activation, Eq. (7)
      nn.h[ℓ][i] = h
      nn.ξ[ℓ][i] = sigmoid(h)
    end
  end

  # copy activation in output layer as output, Eq. (9)
  y_out .= nn.ξ[nn.L]
end


function back_propagate!(nn::NeuralNet, y_in::Vector{Float64}, z::Vector{Float64})
  # reversed order l = L, L−1, L−2, ..., 2
  
  # compute values in the output layers, Eq. (11)
  for i in 1:nn.n[nn.L]
    nn.Δ[nn.L][i] = sigmoid_derivative(nn.h[nn.L][i]) * (y_in[i] - z[i])
  end

  # back-propagate to the rest of the network, Eq. (12)
  for ℓ in reverse(2:nn.L)
    for j in 1:nn.n[ℓ - 1]
      m = 0
      for i in 1:nn.n[ℓ]
        m += nn.Δ[ℓ][i] * nn.w[ℓ][i, j]
      end
      nn.Δ[ℓ - 1][j] = sigmoid_derivative(nn.h[ℓ - 1][j]) * m
    end  
  end
end


function update_weights_thresholds!(nn::NeuralNet, η::Float64, α::Float64)
  # calculate the modification of all the weights and thresholds, Eq. (14)
  for ℓ in reverse(2:nn.L)
    for i in 1:nn.n[ℓ]
      for j in 1:nn.n[ℓ - 1]
        nn.δw[ℓ][i, j] = (-1) * η * nn.Δ[ℓ][i] * nn.ξ[ℓ - 1][j] + α * (if nn.δwᵖʳᵉᵛ[ℓ][i, j] === nothing 0 else nn.δwᵖʳᵉᵛ[ℓ][i, j] end)
        nn.δwᵖʳᵉᵛ[ℓ][i, j] = nn.δw[ℓ][i, j]
        # update weights, Eq. (15)
        nn.w[ℓ][i, j] += nn.δw[ℓ][i, j]
      end
      nn.δθ[ℓ][i] = η * nn.Δ[ℓ][i] + α * (if nn.δθᵖʳᵉᵛ[ℓ][i] === nothing 0 else nn.δθᵖʳᵉᵛ[ℓ][i] end)
      nn.δθᵖʳᵉᵛ[ℓ][i] = nn.δθ[ℓ][i]
      # update thresholds, Eq. (15)
      nn.θ[ℓ][i] += nn.δθ[ℓ][i]
    end
  end
end


function train_nn(nn::NeuralNet, num_epochs::Int64, training_patterns::Matrix{Float64}, η::Float64, α::Float64)
  mat = ["epoch" "E"]
  # Online BP
  num_training_patterns = size(training_patterns, 1)
  for epoch in 1:num_epochs
    # Shuffle the training set
    training_patterns = training_patterns[Random.shuffle(1:end), :]
    for pattern in 1:num_training_patterns
      y_out = [0.0]
      # Feed−forward propagation of pattern x to obtain the output o(x)
      feed_forward!(nn, training_patterns[pattern, 1:nn.n[1]], y_out)
      y_out

      # Back−propagate the error for this pattern
      back_propagate!(nn, y_out, [training_patterns[pattern, end]])

      # Update the weights and thresholds
      update_weights_thresholds!(nn, η, α)
    end

    # Feed−forward all training patterns and calculate their prediction quadratic error
    z = 0.0
    sum_of_variance = 0.0
    for pattern in 1:num_training_patterns
      y_out = [0.0]
      # Feed−forward propagation of pattern x to obtain the output o(x)
      feed_forward!(nn, training_patterns[pattern, 1:nn.n[1]], y_out)
      y_out 

      z = training_patterns[pattern, end]  # desired output z
      
      #println("Desired output: ", z, "    Prediction: ", y_out[1])
      sum_of_variance += abs2(y_out[1] - z)
    end
    # calculate the prediction quadratic error
    E = (1/2) * sum_of_variance
    
    #println("Epoch: ", epoch, "\tPrediction quadratic error: ", E)
    mat = [mat;[epoch E]]
  end
  writedlm("output/output_epoch_and_prediction_quadratic_error.txt", mat)
end


function test_nn(nn::NeuralNet, test_patterns)
  # Feed−forward all test patterns and calculate their prediction percentage error
  mat = ["desired_output" "prediction"]
  sum_of_variance = 0.0
  sum_of_desired_output = 0.0
  num_test_patterns = size(test_patterns, 1)
  for pattern in 1:num_test_patterns
    y_out = [0.0]
    # Feed−forward propagation of pattern x to obtain the output o(x)
    feed_forward!(nn, test_patterns[pattern, 1:nn.n[1]], y_out)
    y_out 

    z = test_patterns[pattern, end]  # desired output z
    
    #println("Desired output: ", z, "\tPrediction: ", y_out[1])
    mat = [mat;[z y_out[1]]]

    sum_of_variance += abs(y_out[1] - z)
    sum_of_desired_output += z
  end
  writedlm("output/output_desired_output_and_prediction.txt", mat)
  # calculate the prediction percentage error
  E = 100 * (sum_of_variance/sum_of_desired_output)
  return E
end
