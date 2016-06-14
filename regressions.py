from sklearn import datasets, linear_model

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

def logistic_regression_gradient(weights,
                                 samples,
                                 classifications,
                                 parameters):

    gradient = np.zeros(weights.shape)
    for i in range(samples.shape[0]):
        logistic = 1. / (1. + np.exp(classifications[i] * \
                                     np.dot(weights.T, samples[i])))

        gradient += np.dot(classifications[i] * samples[i], logistic)

    return (-1 / samples.shape[0]) * gradient

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

# Utilities:

def load_file(npz_filename):
    with np.load(npz_filename) as files:
        training_samples         = files['Xtrain'].item(0).toarray()
        training_classifications = files['ytrain']

        testing_samples          = files['Xteste'].item(0).toarray()
        testing_classifications  = files['yteste']

    return [training_samples, training_classifications,
            testing_samples, testing_classifications]

def get_error_percentage(predictions, testing_classifications):
    errors = 0
    for i, j in zip(predictions, testing_classifications):
        if np.sign(i) != np.sign(j):
            errors += 1

    return float(errors) / float(len(testing_classifications))

def get_accuracy_percentage(predictions, testing_classifications):
    return 1. - get_error_percentage(predictions, testing_classifications)

# Gradient Descent:

def get_initial_weights(size):
    return np.zeros(size)

def gradient_descent(training_samples,
                     training_classifications,
                     testing_samples,
                     iterations               = 10,
                     gradient_function        = linear_regression_gradient,
                     gradient_parameters      = None,
                     learning_rate_function   = constant_learning_rate,
                     learning_rate_parameters = {'rate': .1},
                     regularizer_function     = no_regularization,
                     regularizer_parameters   = None,
                     batch_percentage         = 1.):

    weights    = get_initial_weights(training_samples.shape[1])
    batch_size = int(batch_percentage * training_samples.shape[0])
    for i in range(iterations):
        batch      = np.random.randint(training_samples.shape[0],
                                       size = batch_size)

        samples         = training_samples[batch, :]
        classifications = training_classifications[batch]

        gradient = gradient_function(weights,
                                     samples,
                                     classifications,
                                     gradient_parameters)

        regularized_gradient = regularizer_function(weights,
                                                    gradient,
                                                    regularizer_parameters)

        learning_rate  = learning_rate_function(i, learning_rate_parameters)
        weights       += -1 * learning_rate * regularized_gradient

    predictions = np.dot(testing_samples, weights)
    return predictions

# Scikit Learn Learners:

def scikit_regression(training_samples,
                      training_classifications,
                      testing_samples,
                      model = "linear",
                      batch_percentage = .01):

    batch_size = int(batch_percentage * training_samples.shape[0])
    batch = np.random.randint(training_samples.shape[0],
                                   size = batch_size)

    samples = training_samples[batch, :]
    classifications = training_classifications[batch]

    if model == "linear":
        regr = linear_model.LinearRegression()
    elif model == "linear_l2":
        regr = linear_model.Ridge()

    regr.fit(samples, classifications)
    predictions = regr.predict(testing_samples)

    return predictions
