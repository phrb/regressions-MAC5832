import numpy as np
import random
import csv
import re

# Regression Methods:

def linear_regression_gradient(weights,
                               samples,
                               classifications,
                               parameters):
    return ((2. / float(samples.shape[0])) * \
            (np.dot(samples.T, np.dot(samples, weights)) - \
             np.dot(samples.T, classifications)))

# Learning Rates:

def constant_learning_rate(iteration,
                           parameters):
    return parameters['rate']

def inverse_log_learning_rate(iteration,
                              parameters):
    return parameters['rate']/(1 + np.log2(iteration + 1))

# Regularization Methods:

def no_regularization(weights,
                      gradient,
                      parameters):
    return gradient

def l2_regularization(weights,
                      gradient,
                      parameters):
    return gradient + (parameters['lambda'] * weights)

# Gradient Descent:

def get_initial_weights(size):
    return np.zeros(size)

def gradient_descent(npz_filename,
                     iterations               = 10,
                     gradient_function        = linear_regression_gradient,
                     gradient_parameters      = None,
                     learning_rate_function   = constant_learning_rate,
                     learning_rate_parameters = {'rate': .1},
                     regularizer_function     = no_regularization,
                     regularizer_parameters   = None,
                     batch_percentage         = 1.):

    with np.load(npz_filename) as files:
        training_samples         = files['Xtrain'].item(0).toarray()
        training_classifications = files['ytrain']

        testing_samples          = files['Xteste'].item(0).toarray()
        testing_classifications  = files['yteste']

    print(training_samples.shape)
    weights = get_initial_weights(training_samples.shape[1])

    batch_size = int(batch_percentage * training_samples.shape[0])
    print(batch_size)
    for i in range(iterations):
#        print("Iteration: {0}".format(i))
#        print("Calculating New Batch...")
        batch      = np.random.randint(training_samples.shape[0],
                                       size = batch_size)

        samples         = training_samples[batch, :]
        classifications = training_classifications[batch]

#        print("Done.\nCalculating Gradient...")
        gradient = gradient_function(weights,
                                     samples,
                                     classifications,
                                     gradient_parameters)

#        print("Done.\nRegularizing Gradient...")
        regularized_gradient = regularizer_function(weights,
                                                    gradient,
                                                    regularizer_parameters)

#        print("Done.\nAdjusting Learning Rate...")
        learning_rate  = learning_rate_function(i, learning_rate_parameters)
#        print("Done.\nUpdating Weights...")
        weights       += -1 * learning_rate * regularized_gradient
#        print("Done.")

    predictions = np.dot(testing_samples, weights)
    errors = 0
    for i, j in zip(predictions, testing_classifications):
        if np.sign(i) != np.sign(j):
            errors += 1

    print(errors, len(testing_classifications))
    return weights

gradient_descent("dataset-tarefa2.npz",
                 learning_rate_function   = inverse_log_learning_rate,
                 learning_rate_parameters = {'rate': 1.},
                 iterations               = 60,
                 batch_percentage         = 1.,
                 regularizer_function     = l2_regularization,
                 regularizer_parameters   = {'lambda': 0.01})
