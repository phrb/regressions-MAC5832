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
            'size'   : 20}

    mpl.rc('font', **font)

def plot_sct(data_x,
             data_y,
             data_error_y,
             plot_name,
             title,
             xlabel,
             ylabel):
    fig     = plt.figure(1, figsize=(9, 6))
    ax      = fig.add_subplot(111)

    ax.scatter(data_x, data_y)

    ax.errorbar(data_x, data_y, yerr = data_error_y, linestyle="None")

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

if __name__ == '__main__':
    data = load_file("dataset-tarefa2.npz")

    training_samples         = data[0]
    training_classifications = data[1]
    testing_samples          = data[2]
    testing_classifications  = data[3]

    data_x = []
    data_y = []
    data_y_error = []

    measurements = 10

    for iterations in range(1000, 20000, 1000):
        print("Iterations: {0}".format(iterations))
        data_ys = []
        for j in range(measurements):
            predictions = gradient_descent(training_samples,
                                           training_classifications,
                                           testing_samples,
                                           gradient_function        = linear_regression_gradient,
                                           learning_rate_function   = inverse_log_learning_rate,
                                           learning_rate_parameters = {'rate': .1},
                                           iterations               = iterations,
                                           batch_percentage         = .0005,
                                           regularizer_function     = l2_regularization,
                                           regularizer_parameters   = {'lambda': .0051})

            data_ys.append(get_accuracy_percentage(predictions, testing_classifications))

        data_y_error.append(stats.sem(data_ys))
        data_y.append(np.mean(data_ys))
        data_x.append(iterations)

    plot_sct(data_x,
             data_y,
             data_y_error,
             "acc_vs_iterations",
             "Acc. vs. Iterations",
             "Iteration",
             "Accuracy")

    predictions = scikit_regression(training_samples,
                                    training_classifications,
                                    testing_samples,
                                    model = "linear_l2",
                                    batch_percentage = .0004)

    print(get_accuracy_percentage(predictions, testing_classifications))
