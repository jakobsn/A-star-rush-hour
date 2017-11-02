#!/usr/bin/python

import numpy as np

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
    return data

def euclidian_distance(x, y):
    #TODO
    print("ED")
#    print(input_vector)
    print(x, y)
    return 1

def exponential_decay(value, constant, time):
    #TODO
    return