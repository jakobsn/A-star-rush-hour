#!/usr/bin/py   thon

import numpy as np
from math import exp

def readTSP(targetFile):
    with open(targetFile) as file:
        data = []
        lines = 0
        for line in file:
            row = []
            if not lines == 0 and not lines == 1:
                coords = line.replace('\n', '').split(' ')[1:]
                for coord in coords:
                    row.append(float(coord))
                data.append(np.array(row))
            lines += 1
        # Remove empty lines
        while not len(data[-1]):
            data.pop()
    return scale_min_max(data, np.amin(data), np.amax(data))

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

def linearDecay(t, valAtZero, time_constant):
    return valAtZero * 1/(t/time_constant)

def exponentialDecay(t, valAtZero, time_constant):
    return valAtZero*exp(-(t/time_constant))

def powerDecay(t, valAtZero, time_constant):
    if t == 0:
        return valAtZero
    else:
        return valAtZero*valAtZero**(t/time_constant)