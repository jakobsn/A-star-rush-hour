#!/usr/bin/py   thon

import numpy as np
from math import exp, sqrt
from tensorflow.examples.tutorials.mnist import input_data
import math
import random


def readTSP(targetFile, scale='avdev'):
    with open(targetFile) as file:
        data = []
        lines = 0
        for line in file:
            row = []
            if not lines == 0 and not lines == 1 and not lines == 2 and not lines == 3 and not lines == 4:
                coords = line.replace('\n', '').split(' ')[1:]
                if coords == 'EOF':
                    break
                for coord in coords:
                    if(len(coord)):
                        row.append(float(coord))
                data.append(np.array(row))
            lines += 1
        # Remove empty lines
        while not len(data[-1]):
            data.pop()
    if scale == 'avgdev':
        return scale_average_and_deviation(data)
    elif scale == 'minmax':
        return scale_min_max(data, np.amin(data), np.amax(data))
    else:
        return data


# Scale data by average and standard deviation
def scale_average_and_deviation(data):
    features = get_data_features(data)
    averages, deviations = calculate_average_and_standard_deviation(features)
    for r, row in enumerate(data):
        for f, feature in enumerate(row[0]):
            data[r][0][f] = ((data[r][0][f]-averages[f])/deviations[f])
    return data


# Calculates average and standard deviation, followed wikipedia: https://simple.wikipedia.org/wiki/Standard_deviation
def calculate_average_and_standard_deviation(features):
    deviations = []
    averages = []
    for feature in features:
        # Calculate average and standard deviation for each feature
        averages.append(sum(feature)/len(feature))
        squared_differences = []
        for difference in feature:
            squared_differences.append((difference-averages[-1])**2)
        deviations.append(sqrt((sum(squared_differences)/len(squared_differences))))
    return averages, deviations


# Get all the input data stored in lists of features
def get_data_features(data):
    # Create list to store lists of input features
    features = []
    for x in range(len(data[0][0])):
        features.append([])

    # Append all features
    for y in range(len(data)):
        for x in range(len(data[y][0])):
            features[x].append(data[y][0][x])
    return features


def euclidian_distance(x, y):
    #TODO
    print("ED")
#    print(input_vector)
    print(x, y)
    return 1


# Scale all input features by the min and max value
def scale_min_max(data, min_feature, max_feature):
    print(data)
    for row in data:
        for i, feature in enumerate(row):
            row[i] = ((feature - min_feature) / (max_feature - min_feature))
    print(data)
    return data


def linearDecay(t, valAtZero, time_constant, epochs):
    return valAtZero * 1/(t/time_constant)


def exponentialDecay(t, valAtZero, time_constant, epochs):
    return valAtZero*exp(-(t/time_constant))


def powerDecay(t, valAtZero, time_constant, epochs):
    if t == 0:
        return valAtZero
    else:
        return valAtZero*time_constant**(t/epochs)


def get_mnist_data(size):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    features, labels = mnist.train.next_batch(size)
    output = []
    for i in range(len(features)):
        output.append(np.array([np.array(features[i].tolist()), np.array(labels[i].tolist())]))
    return output


def get_mnist_test_data(size=100):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    features, labels = mnist.test.next_batch(size)
    output = []
    for i in range(len(features)):
        output.append(np.array([np.array(features[i].tolist()), np.array(labels[i].tolist())]))
    return output


def generate_points(center_x, center_y, mean_radius, sigma_radius, num_points):
    points = [[], []]
    for theta in np.linspace(0, 2 * math.pi - (2 * math.pi / num_points), num_points):
        radius = random.gauss(mean_radius, sigma_radius)
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        points[0].append(x)
        points[1].append(y)
    npoints = np.array(points)
    return npoints