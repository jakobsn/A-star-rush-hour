#!/usr/bin/python

import numpy as np
import somtools as st
import numpy_indexed as npi
from math import sqrt, ceil, floor, exp
import six.moves.cPickle as cPickle


class SOM:
    def __init__(self, epochs, lrate, hoodsize, insize, outsize, features, radius, weight_range, lrate_decay, hood_decay, topo):
        self.epochs = epochs
        self.features = features
        np.random.shuffle(self.features)
        self.radius = radius
        self.weight_range = weight_range
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.initial_lrate = lrate
        self.hoodsize = hoodsize
        self.hood_decay = hood_decay
        self.initial_hood = hoodsize
        self.global_training_step = 0
        self.topo = topo
        if insize is None or outsize is None:
            self.insize = len(features[0])
            self.outsize = len(features)
        else:
            self.insize = insize
            self.outsize = outsize
        self.weights = self.initial_weights()
        self.neuronRing = self.create_neuron_ring()
        self.i = 0 # Keep track of which weight being changed for input neuron
        self.do_training()


    def do_training(self):
        # TODO: Find winning neuron for each vector
        #       Update weights for winner and neighbours
        #       Update lrate and hoodsize
        for epoch in range(self.epochs):
            np.random.shuffle(self.features)
            self.neuronRing = self.create_neuron_ring()
            for feature in self.features:
                distances, min_distance, winner_neuron = self.findWinner(feature)
                neighbours = self.get_neighbours(winner_neuron)
                self.neuronRing[winner_neuron] += 1
                self.adjust_clusters(neighbours)
            self.hoodsize = ceil(self.hood_decay(epoch, self.initial_hood, 1))
            self.lrate = self.lrate_decay(epoch, self.initial_lrate, 1)
            print("hood, lrate", self.hoodsize, self.lrate)
        return

    def findWinner(self, feature):
        eDistance = np.vectorize(self.euclidian_distance, cache=True)
        self.input_vector = feature
        distances = eDistance(*self.weights)
        min_distance = np.min(distances)
        winner_neuron = np.argmin(distances)
        return distances, min_distance, winner_neuron

    def adjust_clusters(self, neighbours):
        adjust_cluster = np.vectorize(self.adjust_cluster, cache=True)
        adjust_cluster(neighbours[0], neighbours[1], self.weights[:, neighbours[0]])
        return

    def adjust_cluster(self, index, hood, weight):
        if self.i < (len(self.input_vector)-1):
            self.i += 1
        else:
            self.i = 0
        self.weights[0][index] = weight + self.lrate*hood*np.subtract(self.input_vector[self.i], weight)
        #print("index", index, "input_value", self.input_vector[self.i], "hood", hood, "weight", weight, "new weight", self.weights[0][index])

    # Get weight indexes
    def get_neighbours(self, winner_neuron):
        if self.topo == "ring":
            return np.array(self.get_ring_neighbours(winner_neuron))
        else:
            # TODO: matrix topology
            pass
        return

    # Return neighbour indices with degree of neighbourhood
    def get_ring_neighbours(self, winner_neuron):
        #TODO
        ring_neighbours = [[], []]
        for n, i in zip(range(winner_neuron-self.hoodsize, winner_neuron+self.hoodsize+1), range(floor(-self.hoodsize), ceil(winner_neuron+self.hoodsize+1))):
            if n < 0:
                ring_neighbours[0].append(n + self.outsize)
            elif n >= self.outsize:
                ring_neighbours[0].append(n - self.outsize)
            else:
                ring_neighbours[0].append(n)

            ring_neighbours[1].append(abs(i))
        return np.array(ring_neighbours)

    def euclidian_distance(self, *weights):
        return np.sqrt(np.sum(np.power(np.subtract(self.input_vector, np.array(weights)), 2)))

    def run_one_step(self):
        #TODO
        return

    def create_neuron_ring(self):
        #TODO: Is that all?
        return np.zeros(shape=[self.outsize])

    # Weight matrix of zeros
    def initial_weights(self):
        return np.random.uniform(self.weight_range[0], self.weight_range[1], [self.insize, self.outsize])


    def do_mapping(self):
        #TODO
        return

#class SOMModule:
#    def __init__(self, som, index, input, ):


#print(st.readTSP('../data/1.txt'))
def main(data_funct=st.readTSP, data_params=('../data/6.txt',), epochs=1000,  lrate=0.1, hoodsize=4, insize=None, outsize=None, radius=1,
         weight_range=[10,40], lrate_decay=st.exponentialDecay, hood_decay=st.exponentialDecay, topo='ring'):
    features =  data_funct(*data_params)
    som = SOM(epochs=epochs, lrate=lrate, hoodsize=hoodsize, features=features, insize=insize, outsize=outsize, radius=radius,
              weight_range=weight_range, lrate_decay=lrate_decay, hood_decay=hood_decay, topo=topo)
    print("features", som.features)
    print(som.neuronRing)
main()

"""
TODO:
- Visualize at step k with total length
- Neighborhood decay
- lrate decay
- TOPOGRAPHY
"""