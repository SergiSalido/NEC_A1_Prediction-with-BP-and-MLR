########################################
# Source code of Back-Propagation (BP) #
########################################


# Import Packages
import Pkg;

# Install packages
Pkg.add(["CSV", "DataFrames", "Plots", "StatsBase", "ScikitLearn"])

# Load the installed packages
using CSV   # library to read the data
using DataFrames
using Plots
using StatsBase
using Random
using DelimitedFiles
using ScikitLearn
plotly()

include("NeuralNet.jl")


# Training parameters
struct Configuration
    input_data_file_name::String            # Name of the data file
    num_training_patterns::Int64            # Number of training patterns
    num_test_patterns::Int64                # Number of test patterns
    num_folds::Int64                        # Number of folds (cross-validation)
    num_layers::Int64                       # Number of layers
    num_units::Vector{Int64}                # Number of units in each layer
    num_epochs::Int64                       # Number of epochs
    η::Float64                              # Learning rate
    α::Float64                              # Momentum
    scaling_method::String                  # Scaling method (normalization or standardization) of inputs and/or outputs
    normalization_range::Vector{Float64}    # Range of the normalized data (in the case of normalization)
    activation_function::String             # Activation function (sigmoid, tanh, ReLU, etc.)
    output_data_file_name::String           # Name of output file(s)
end


function Configuration(parameters_file_name::String)
    # load training parameters
    lines = readlines(parameters_file_name)
    
    input_data_file_name = lines[1]
    num_training_patterns = parse(Int64, split(lines[2],';')[1])
    num_test_patterns = parse(Int64, split(lines[2],';')[2])
    num_folds = parse(Int64, lines[3])
    num_layers = parse(Int64, lines[4])
    num_units = Vector{Int64}()
    num_units_string = split(lines[5],';')
    for i in 1:length(num_units_string)
        push!(num_units, parse(Int64, num_units_string[i]))
    end
    num_epochs = parse(Int64, lines[6])
    η = parse(Float64, split(lines[7],';')[1])  # between 0.2 and 0.01
    α = parse(Float64, split(lines[7],';')[2])  # between 0.0 (no momentum at all) and 0.9
    scaling_method = lines[8]
    normalization_range = Float64[parse(Float64, split(lines[9],';')[1]), parse(Float64, split(lines[9],';')[2])]
    activation_function = lines[10]
    output_data_file_name = lines[11]

    return Configuration(input_data_file_name, num_training_patterns, num_test_patterns, num_folds, num_layers, num_units, num_epochs, η, α, scaling_method, normalization_range, activation_function, output_data_file_name)
end


function standardization(m::Matrix{Float64})    # scaling method
    dt = StatsBase.fit(ZScoreTransform, m, dims=1)
    normal = StatsBase.transform(dt, m)
	return normal
end


function normalization(m::Matrix{Float64}, lower::Float64, upper::Float64)    # scaling method
    for col in eachcol(m)
        xmin = minimum(col)
        xmax = maximum(col)
        n = length(col)
        for i in 1:n
            col[i] = lower + (col[i] - xmin) / (xmax - xmin) * (upper - lower)
        end
    end
    return m
end




if size(ARGS)==1
    @info "Getting the parameters."
    cfg = Configuration(ARGS[1])
else
    @error "ERROR: Unknown ARGS. Name of the parameters file is missing or there is more than one argument."
    @info "Getting the parameters of parameters_file.txt."
    cfg = Configuration("parameters_file.txt")
    #println(cfg)
end


# Import dataset
df = DataFrame(CSV.File(cfg.input_data_file_name))
data = Matrix(df)


# Scale input and output patterns
if cfg.scaling_method == "normalization"
    lower = cfg.normalization_range[1]
    upper = cfg.normalization_range[2]
    df_scaled = DataFrame(normalization(data, lower, upper), :auto)
elseif cfg.scaling_method == "standardization"
    df_scaled = DataFrame(standardization(data), :auto)
end


# Split between training set and test set
training_patterns = first(df_scaled, cfg.num_training_patterns)
test_patterns = last(df_scaled, cfg.num_test_patterns)


# Shuffle
training_patterns = training_patterns[Random.shuffle(1:end), :]
#println(training_patterns)


# Convert to Matrix
training_patterns = Matrix(training_patterns)
test_patterns = Matrix(test_patterns)


# K-fold cross-validation
println("K-fold cross-validation")
kf = ScikitLearn.CrossValidation.KFold(cfg.num_training_patterns, n_folds=cfg.num_folds)
prediction_errors_of_validations = Vector{Float64}()
for (training_index, test_index) in kf
    #println("training_index:\n", training_index)
    #println("test_index:\n", test_index)

    nn = NeuralNet(cfg.num_units)

    # Training (use k-1 subsets to train a NN)
    train_nn(nn, cfg.num_epochs, training_patterns[training_index, :], cfg.η, cfg.α)
    df_output = DataFrame(CSV.File("output/output_epoch_and_prediction_quadratic_error.txt"))
    scatter(df_output.epoch, df_output.E, xlab="epoch", ylab="E")

    # Validation (predict over the remaining validation subset)
    validation_average_prediction_error = test_nn(nn, training_patterns[test_index, :])
    println("Fold ", "\tAverage prediction error of validation subset is ", validation_average_prediction_error)

    # Save the average prediction error to a vector
    push!(prediction_errors_of_validations, validation_average_prediction_error)
end
# Calculate the average of the prediction errors for all the validation sets
cv_average_prediction_error = sum(prediction_errors_of_validations) / cfg.num_folds
println("The average prediction error of cross-validation is ", cv_average_prediction_error)



# Train a NN with "all" the training patterns
println("\nTraining a NN with 'all' the training patterns...")
nn = NeuralNet(cfg.num_units)
train_nn(nn, cfg.num_epochs, training_patterns, cfg.η, cfg.α)
# Write weights and thresholds of the trained network to text files
writedlm("output/nn_weights"*cfg.output_data_file_name*".txt", nn.w)
writedlm("output/nn_thresholds"*cfg.output_data_file_name*".txt", nn.θ)
# Plot the evolution of the training error
df_output = DataFrame(CSV.File("output/output_epoch_and_prediction_quadratic_error.txt"))
scatter(df_output.epoch, df_output.E, xlab="epoch", ylab="E")



# Test the NN with the testing patterns
println("\nTesting the NN with the testing patterns...")
E = test_nn(nn, test_patterns)
println("The prediction percentage error is ", E)
# Plot the evolution of the test error
df_output = DataFrame(CSV.File("output/output_epoch_and_prediction_quadratic_error.txt"))
scatter(df_output.epoch, df_output.E, xlab="epoch", ylab="E")
# Scatter plot of the prediction versus real value
df_output = DataFrame(CSV.File("output/output_desired_output_and_prediction.txt"))
scatter(df_output.desired_output, df_output.prediction, xlab="desired_output", ylab="prediction")

#TODO: Descale the predictions of test patterns, and evaluate them
