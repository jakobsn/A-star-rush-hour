#!/usr/bin/python

import numpy as np
import somtools as st
import numpy_indexed as npi
from math import sqrt, ceil
import six.moves.cPickle as cPickle


class SOM:
    def __init__(self, epochs, lrate, insize, outsize, features, radius, weight_range, lrate_decay, hood_decay):
        self.epochs = epochs
        self.lrate = lrate
        self.features = features
        np.random.shuffle(self.features)
        self.radius = radius
        self.weight_range = weight_range
        self.lrate_decay = lrate_decay
        self.hood_decay = hood_decay
        self.global_training_step = 0
        if insize is None or outsize is None:
            self.insize = len(features[0])
            self.outsize = len(features)
        else:
            self.insize = insize
            self.outsize = outsize
        self.weights = self.initial_weights()
        self.weights2 = cPickle.loads(cPickle.dumps(self.weights, -1))
        self.weights2 = np.rot90(self.weights2)
        self.neuronRing = self.create_neuron_ring()
        print("inout", self.insize, self.outsize)
        print("weights")
        print(self.weights)
        #print("weights2")
        #print(self.weights2)
        print("ring")
        print(self.neuronRing)
        self.do_training()


    def do_training(self):
        # TODO: Find winning neuron for each vector
        # TODO: Update weights for winner and neighbours
        eDistance = np.vectorize(self.euclidian_distance)
        for epoch in range(self.epochs):
            for feature in self.features:
                self.input_vector = feature
                e = eDistance(self.weights[0], self.weights[1])
                print("e", e)
                break
            break
            self.features = np.random.shuffle(self.features)
        return

    def euclidian_distance(self, x, y):
        # TODO
        print("ed", self.input_vector, x, y)
        return 1

    def run_one_step(self):
        #TODO
        return

    def create_neuron_ring(self):
        #TODO: Is that all?
        return np.zeros(shape=[self.outsize])

    def initial_weights(self):
        return np.random.uniform(-0.1, 0.1, [self.insize, self.outsize])

    def adjust_weigths(self):
        #TODO
        return

    def do_mapping(self):
        #TODO
        return

    def findWinner(self):
        #TODO
        return

#class SOMModule:
#    def __init__(self, som, index, input, ):


#print(st.readTSP('../data/1.txt'))
def main(data_funct=st.readTSP, data_params=('../data/6.txt',), epochs=100,  lrate=0.1, insize=None, outsize=None, radius=1,
         weight_range=[-0.1,0.1], lrate_decay=0, hood_decay=0):
    features =  data_funct(*data_params)
    som = SOM(epochs=epochs, lrate=lrate, features=features, insize=insize, outsize=outsize, radius=radius, weight_range=weight_range,
              lrate_decay=lrate_decay, hood_decay=hood_decay)
    print("features", som.features)

main()

"""
TODO:
- Visualize at step k with total length
- Neighborhood decay
- lrate decay
- TOPOGRAPHY
"""