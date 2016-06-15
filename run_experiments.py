#! /usr/bin/python

from regressions import *

from scipy import stats

import os
import matplotlib as mpl

mpl.use('agg')

import matplotlib.pyplot as plt

def config_matplotlib():
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    font = {'family' : 'serif',
            'size'   : 18}

    mpl.rc('font', **font)

def plot_sct(data_x,
             data_y,
             data_error_y,
             plot_name,
             title,
             xlabel,
             ylabel):
    fig = plt.figure(1, figsize=(9, 6))
    ax  = fig.add_subplot(111)

    ax.scatter(data_x, data_y)
    ax.errorbar(data_x, data_y, yerr = data_error_y, linestyle="None")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    fig.savefig("{0}.eps".format(plot_name), format = 'eps', dpi = 1000)

    plt.clf()

def plot_sct_cmp(data1_x,
                 data1_y,
                 data1_error_y,
                 data2_x,
                 data2_y,
                 data2_error_y,
                 plot_name,
                 title,
                 xlabel,
                 ylabel):
    fig = plt.figure(1, figsize=(9, 6))
    ax  = fig.add_subplot(111)

    ax.scatter(data1_x, data1_y)
    ax.errorbar(data1_x, data1_y, yerr = data1_error_y, linestyle="None")

    ax.scatter(data2_x, data2_y)
    ax.errorbar(data2_x, data2_y, yerr = data2_error_y, linestyle="None")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    fig.savefig("{0}.eps".format(plot_name), format = 'eps', dpi = 1000)

    plt.clf()

def plot_bar(index_range,
             data,
             plot_name,
             title,
             xlabel,
             ylabel,
             tick_labels):
    fig     = plt.figure(1, figsize=(9, 6))
    ax      = fig.add_subplot(111)

    indexes = np.arange(index_range)
    width   = 0.5

    ax.bar(indexes, data, width, color='black')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xticks(indexes + (width / 2))
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(tick_labels, rotation = 30)

    plt.tight_layout()

    fig.savefig("{0}.eps".format(plot_name), format = 'eps', dpi = 1000)

    plt.clf()

def measure_iterations(training_samples,
                       training_classifications,
                       testing_samples,
                       testing_classifications,
                       gradient,
                       rate,
                       rate_parameters,
                       batch,
                       regularizer,
                       regularizer_parameters,
                       file_title,
                       title):
    data_x = []
    data_y = []
    data_y_error = []

    measurements = 10

    for iterations in range(1000, 30000, 1000):
        print("Iterations: {0}".format(iterations))
        data_ys = []
        for j in range(measurements):
            predictions = gradient_descent(training_samples,
                                           training_classifications,
                                           testing_samples,
                                           gradient_function        = gradient,
                                           learning_rate_function   = rate,
                                           learning_rate_parameters = rate_parameters,
                                           iterations               = iterations,
                                           batch_percentage         = batch,
                                           regularizer_function     = regularizer,
                                           regularizer_parameters   = regularizer_parameters)

            data_ys.append(get_accuracy_percentage(predictions, testing_classifications))

        data_y_error.append(stats.sem(data_ys))
        data_y.append(np.mean(data_ys))
        data_x.append(iterations)

    plot_sct(data_x,
             data_y,
             data_y_error,
             "acc_vs_iterations_{0}".format(file_title),
             "Acc. vs. Iterations {0}".format(title),
             "Iteration",
             "Accuracy")

def measure_all_iterations(training_samples,
                           training_classifications,
                           testing_samples,
                           testing_classifications):
    measure_iterations(training_samples,
                       training_classifications,
                       testing_samples,
                       testing_classifications,
                       linear_regression_gradient,
                       inverse_log_learning_rate,
                       {'rate': .5},
                       .0005,
                       no_regularization,
                       {'lambda': .0051}, # Does not matter here
                       "linreg",
                       "(Linear Regression)")
    measure_iterations(training_samples,
                       training_classifications,
                       testing_samples,
                       testing_classifications,
                       linear_regression_gradient,
                       inverse_log_learning_rate,
                       {'rate': .5},
                       .0005,
                       l2_regularization,
                       {'lambda': .0051},
                       "linregL2",
                       "(Linear Regression with L2)")
    measure_iterations(training_samples,
                       training_classifications,
                       testing_samples,
                       testing_classifications,
                       logistic_regression_gradient,
                       inverse_log_learning_rate,
                       {'rate': 2.},
                       .0005,
                       no_regularization,
                       {'lambda': .0051}, # Does not matter here
                       "logreg",
                       "(Logistic Regression)")
    measure_iterations(training_samples,
                       training_classifications,
                       testing_samples,
                       testing_classifications,
                       linear_regression_gradient,
                       inverse_log_learning_rate,
                       {'rate': 2.},
                       .0005,
                       l2_regularization,
                       {'lambda': .0051}, # Does not matter here
                       "logregL2",
                       "(Logistic Regression with L2)")

def measure_rate(training_samples,
                 training_classifications,
                 testing_samples,
                 testing_classifications,
                 gradient,
                 iterations,
                 rate_function,
                 batch,
                 regularizer,
                 regularizer_parameters,
                 file_title,
                 title):
    data_x = []
    data_y = []
    data_y_error = []

    measurements = 5
    rates        = 25
    rate         = 32.

    for k in range(rates):
        print("Rate: {0}".format(rate))

        data_ys = []
        for j in range(measurements):
            predictions = gradient_descent(training_samples,
                                           training_classifications,
                                           testing_samples,
                                           gradient_function        = gradient,
                                           learning_rate_function   = rate_function,
                                           learning_rate_parameters = {'rate': rate},
                                           iterations               = iterations,
                                           batch_percentage         = batch,
                                           regularizer_function     = regularizer,
                                           regularizer_parameters   = regularizer_parameters)

            data_ys.append(get_accuracy_percentage(predictions, testing_classifications))

        data_y_error.append(stats.sem(data_ys))
        data_y.append(np.mean(data_ys))
        data_x.append(np.log2(rate))

        rate /= 2.

    plot_sct(data_x,
             data_y,
             data_y_error,
             "acc_vs_rate_{0}".format(file_title),
             "Accuracy. vs. Learning Rate {0}".format(title),
             "Learning Rate (log2)",
             "Accuracy")

def measure_all_rates(training_samples,
                      training_classifications,
                      testing_samples,
                      testing_classifications):
    measure_rate(training_samples,
                 training_classifications,
                 testing_samples,
                 testing_classifications,
                 linear_regression_gradient,
                 15000,
                 inverse_log_learning_rate,
                 .0005,
                 no_regularization,
                 {'lambda': .0051}, # Does not matter here
                 "linreg",
                 "(Linear Regression)")

    measure_rate(training_samples,
                 training_classifications,
                 testing_samples,
                 testing_classifications,
                 linear_regression_gradient,
                 15000,
                 inverse_log_learning_rate,
                 .0005,
                 l2_regularization,
                 {'lambda': .0051}, # Does not matter here
                 "linregL2",
                 "(Linear Regression with L2)")

    measure_rate(training_samples,
                 training_classifications,
                 testing_samples,
                 testing_classifications,
                 logistic_regression_gradient,
                 15000,
                 inverse_log_learning_rate,
                 .0005,
                 no_regularization,
                 {'lambda': .0051}, # Does not matter here
                 "logreg",
                 "(Logistic Regression)")

    measure_rate(training_samples,
                 training_classifications,
                 testing_samples,
                 testing_classifications,
                 logistic_regression_gradient,
                 15000,
                 inverse_log_learning_rate,
                 .0005,
                 l2_regularization,
                 {'lambda': .0051}, # Does not matter here
                 "logregL2",
                 "(Logistic Regression with L2)")

def measure_batch(training_samples,
                  training_classifications,
                  testing_samples,
                  testing_classifications,
                  gradient,
                  iterations,
                  rate,
                  rate_function,
                  regularizer,
                  regularizer_parameters,
                  file_title,
                  title):
    data_x = []
    data_y = []
    data_y_error = []

    measurements = 5
    batches      = 10
    batch        = .0005

    for k in range(batches):
        print("Batch: {0}".format(batch))

        data_ys = []
        for j in range(measurements):
            predictions = gradient_descent(training_samples,
                                           training_classifications,
                                           testing_samples,
                                           gradient_function        = gradient,
                                           learning_rate_function   = rate_function,
                                           learning_rate_parameters = {'rate': rate},
                                           iterations               = iterations,
                                           batch_percentage         = batch,
                                           regularizer_function     = regularizer,
                                           regularizer_parameters   = regularizer_parameters)

            data_ys.append(get_accuracy_percentage(predictions, testing_classifications))

        data_y_error.append(stats.sem(data_ys))
        data_y.append(np.mean(data_ys))
        data_x.append(np.log2(batch))

        batch *= 2.

    plot_sct(data_x,
             data_y,
             data_y_error,
             "acc_vs_batchp_{0}".format(file_title),
             "Accuracy. vs. Batch {0}".format(title),
             "Percentage of Examples (log2)",
             "Accuracy")

def measure_all_batches(training_samples,
                        training_classifications,
                        testing_samples,
                        testing_classifications):
    iterations = 500
    measure_batch(training_samples,
                  training_classifications,
                  testing_samples,
                  testing_classifications,
                  linear_regression_gradient,
                  iterations,
                  .4,
                  inverse_log_learning_rate,
                  no_regularization,
                  {'lambda': .0051}, # Does not matter here
                  "linreg",
                  "(Linear Regression)")

    measure_batch(training_samples,
                  training_classifications,
                  testing_samples,
                  testing_classifications,
                  linear_regression_gradient,
                  iteraitons,
                  .4,
                  inverse_log_learning_rate,
                  l2_regularization,
                  {'lambda': .0051}, # Does not matter here
                  "linregL2",
                  "(Linear Regression with L2)")

    measure_batch(training_samples,
                  training_classifications,
                  testing_samples,
                  testing_classifications,
                  logistic_regression_gradient,
                  iterations,
                  2.,
                  inverse_log_learning_rate,
                  no_regularization,
                  {'lambda': .0051}, # Does not matter here
                  "logreg",
                  "(Logistic Regression)")

    measure_batch(training_samples,
                  training_classifications,
                  testing_samples,
                  testing_classifications,
                  logistic_regression_gradient,
                  iterations,
                  2.,
                  inverse_log_learning_rate,
                  l2_regularization,
                  {'lambda': .0051}, # Does not matter here
                  "logregL2",
                  "(Logistic Regression with L2)")

if __name__ == '__main__':
    config_matplotlib()
    data = load_file("dataset-tarefa2.npz")

    training_samples         = data[0]
    training_classifications = data[1]
    testing_samples          = data[2]
    testing_classifications  = data[3]

    measure_all_batches(training_samples,
                        training_classifications,
                        testing_samples,
                        testing_classifications)

#    measure_all_rates(training_samples,
#                      training_classifications,
#                      testing_samples,
#                      testing_classifications)

#    measure_all_iterations(training_samples,
#                           training_classifications,
#                           testing_samples,
#                           testing_classifications)

#    predictions = scikit_regression(training_samples,
#                                    training_classifications,
#                                    testing_samples,
#                                    model = "linear_l2",
#                                    batch_percentage = .0004)
#
#    print(get_accuracy_percentage(predictions, testing_classifications))
