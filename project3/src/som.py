#!/usr/bin/python

import numpy as np
import somtools as st
import numpy_indexed as npi
from math import sqrt, ceil, floor, exp
import six.moves.cPickle as cPickle
import matplotlib.pyplot as PLT
from time import sleep, time
from random import randint


class SOM:
    def __init__(self, epochs, lrate, hoodsize, insize, outsize, features,
                 weight_range, lrate_decay, hood_decay, lrConstant,
                 hoodConstant, showint,show_sleep, network_dims=None):
        self.epochs = epochs
        self.features = features
        self.weight_range = weight_range
        self.lrate = lrate
        self.lrate_decay = lrate_decay
        self.initial_lrate = lrate
        self.lrConstant = lrConstant
        self.hoodsize = hoodsize
        self.hood_decay = hood_decay
        self.initial_hood = hoodsize
        self.hoodConstant = hoodConstant
        self.global_training_step = 0
        self.network_dims = network_dims
        if insize is None or outsize is None:
            self.insize = len(features[0])
            self.outsize = len(features)*2
        else:
            self.insize = insize
            self.outsize = outsize
        if self.network_dims is not None:
            self.topo = "matrix"
            self.neuron_matrix = self.create_neuron_matrix()
        else:
            self.topo = "ring"
        self.weights = self.initial_weights()
        self.neuronRing = self.create_neuron_ring()
        self.input_vector = None # Keep track of current input vector
        self.i = 0 # Keep track of which weight being changed for input neuron
        self.isDisplayed = False
        self.showint = showint
        self.show_sleep = show_sleep
        self.distance = self.do_training()

    # Pick random input feature
    # Update weights for winner and neighbours
    # Update lrate and hoodsize
    def do_training(self):
        #print(self.weights)
        self.neuronRing = self.create_neuron_ring()
        for epoch in range(self.epochs):
            if self.lrate == 0:
                break
            if self.topo == "matrix":
                feature = self.features[randint(0, len(self.features)-1)][0]
            else:
                feature = self.features[randint(0, len(self.features)-1)]
            #print("feature", feature)
            distances, min_distance, winner_neuron = self.findWinner(feature)
            neighbours = self.get_neighbours(winner_neuron)
            self.neuronRing[winner_neuron] += 1
            self.adjust_clusters(neighbours)
            self.hoodsize = round(self.hood_decay(epoch, self.initial_hood, self.hoodConstant, self.epochs))
            self.lrate = self.lrate_decay(epoch, self.initial_lrate,self.lrConstant, self.epochs)
            print(str('[' + str(epoch) + ']'), "hood, lrate", self.hoodsize, self.lrate)
            if not self.showint is 0 and (epoch == 0 or epoch % self.showint == 0):
                self.do_mapping(self.weight_range, self.hoodsize, self.lrate, epoch, self.show_sleep)
                print("Neuron ring", self.neuronRing)
        return self.findTotalDistance()

    # Find distance for TSP solution
    def findTotalDistance(self):
        #TODO
        distance = 0
        return distance

    def findWinner(self, feature):
        eDistance = np.vectorize(self.euclidian_distance, cache=True)
        self.input_vector = feature
        #print("weights", self.weights)
        weights = np.rot90(self.weights, 1)[::-1]
        distances = self.euclidian_distance(weights)
        min_distance = np.min(distances)
        winner_neuron = np.argmin(distances)
        return distances, min_distance, winner_neuron

    def adjust_clusters(self, neighbours):
        adjust_cluster = np.vectorize(self.adjust_cluster, cache=True)
        adjust_cluster(neighbours[0], neighbours[1], self.weights[:, neighbours[0]])
        return

    def adjust_cluster(self, index, hood, weight):
        self.weights[self.i][index] = weight + self.lrate*(hood+1)*np.subtract(self.input_vector[self.i], weight)
        #print("index", index, "input_value", self.input_vector[self.i], "hood", hood, "weight", weight, "new weight", self.weights[0][index])
        if self.i < (len(self.input_vector)-1):
            self.i += 1
        else:
            self.i = 0

    # Get weight indexes
    def get_neighbours(self, winner_neuron):
        if self.topo == "ring":
            return self.get_ring_neighbours(winner_neuron)
        else:
            return self.get_matrix_neighbours(winner_neuron)

    # Return neighbour indices with degree of neighbourhood
    def get_matrix_neighbours(self, winner_neuron):
        matrix_neighbours = [[], []]
        broke = False
        for i in range(self.network_dims[0]):
            for j in range(self.network_dims[1]):
                if self.neuron_matrix[i][j] == winner_neuron:
                    winner_y, winner_x = i, j
                    broke = True
                    break
            if broke:
                break

        for i in range(winner_y-self.hoodsize, winner_y + self.hoodsize+1):
            if 0 <= i < self.network_dims[0]:
                for j in range(winner_x-self.hoodsize, winner_x+self.hoodsize+1):
                    if 0 <= j < self.network_dims[1]:
                        hoodsizes = [abs(winner_y - i), abs(winner_x - j)]
                        if not np.isnan(self.neuron_matrix[i][j]):
                            matrix_neighbours[0].append(int(self.neuron_matrix[i][j]))
                            matrix_neighbours[1].append(hoodsizes[np.argmax(hoodsizes)])
        return np.array(matrix_neighbours)

    # Return neighbour indices with degree of neighbourhood
    def get_ring_neighbours(self, winner_neuron):
        ring_neighbours = [[], []]
        for n, i in zip(range(winner_neuron-self.hoodsize, winner_neuron+self.hoodsize+1),
                        range(floor(-self.hoodsize), ceil(winner_neuron+self.hoodsize+1))):
            if n < 0:
                ring_neighbours[0].append(n + self.outsize)
            elif n >= self.outsize:
                ring_neighbours[0].append(n - self.outsize)
            else:
                ring_neighbours[0].append(n)

            ring_neighbours[1].append(abs(i))
        return np.array(ring_neighbours)

    def euclidian_distance(self, weights):
        distances = []
        for weight_vector in weights:
            distances.append(np.sum(np.power(np.subtract(self.input_vector, weight_vector), 2)))
        return distances

    def create_neuron_matrix(self):
        neuron_matrix = np.zeros(self.network_dims)
        neuron_count = 0
        for i in range(self.network_dims[0]):
            for j in range(self.network_dims[1]):
                if neuron_count >= self.outsize:
                    neuron_matrix[i][j] = None
                else:
                    neuron_matrix[i][j] = neuron_count
                    neuron_count += 1
        return neuron_matrix

    def create_neuron_ring(self):
        return np.zeros(shape=[self.outsize])

    # Weight matrix of zeros
    def initial_weights(self):
        return np.random.uniform(self.weight_range[0], self.weight_range[1], [self.insize, self.outsize])

    def do_mapping(self, weight_range=None, hood=None, lrate=None, epochs=None, step='NA', sleep_time=1):
        if hood == None: hood=self.hoodsize
        if lrate == None: lrate=self.lrate
        if epochs == None: epochs=self.epochs
        if weight_range == None: weight_range=self.weight_range
        if self.isDisplayed:
            sleep(sleep_time)
            PLT.close("all")

        fig = PLT.figure()
        # Scatter weights
        for wx, wy in zip(self.weights[0], self.weights[1]):
            PLT.scatter(wx, wy, c="red")
        # Plot weight edges
        PLT.plot(self.weights[0], self.weights[1])
        PLT.plot([self.weights[0][0], self.weights[0][-1]], [self.weights[1][0], self.weights[1][-1]], c='blue')
        # Scatter features
        for feature in self.features:
            PLT.scatter(feature[0], feature[1], c="black")
        fig.suptitle("Run " + str(step) + " Epochs " + str(epochs) + " Lrate " + str(lrate) \
                     + " Hood " + str(hood) + " Weight range " + str(weight_range) + \
                     " Outsize " + str(self.outsize) + " constants " + str(self.lrConstant) + ',' + str(self.hoodConstant))
        PLT.show(block=False)
        self.isDisplayed = True


def main(data_funct=st.readTSP, data_params=('../data/6.txt',), epochs=3000,  lrate=0.2, hoodsize=4,
         insize=2, outsize=60, weight_range=[0.49, 5], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
         lrConstant=0.09, hoodConstant=300, showint=1000, show_sleep=2, final_sleep=20, network_dims=None):
    features = data_funct(*data_params)
    start = time()

    som = SOM(epochs=epochs, lrate=lrate, hoodsize=hoodsize, features=features, insize=insize, outsize=outsize,
              weight_range=weight_range, lrate_decay=lrate_decay, hood_decay=hood_decay, lrConstant=lrConstant,
              hoodConstant=hoodConstant, showint=showint, show_sleep=show_sleep, network_dims=network_dims)
    print('funct', data_funct, 'params', data_params, 'epochs', epochs, 'lrate', lrate, '\n',
          'hoodsize', hoodsize, 'insize', insize, 'outsize', outsize,  'weight_range', weight_range, '\n',
          'lrate_deacay', lrate_decay, 'hood_decay', hood_decay,   '\n',
          'topo', som.topo, "lrConstant", lrConstant, "hoodConstant", hoodConstant, '\n',
          "showint", showint, 'show_sleep', show_sleep, 'final_sleep', final_sleep, 'distance', som.distance)
    print(som.weights)
    end = time()
    print("Time elapsed:", end - start, "s", (end-start)/60, "m")
    print("Neuron ring", som.neuronRing)

    som.do_mapping(weight_range, hoodsize, lrate, epochs, 'Final', final_sleep)


# main(network_dims=[9, 10])


main(data_funct=st.get_mnist_data, data_params=(60,), epochs=10, lrate=0.2, hoodsize=2, insize=784, outsize=10,
     weight_range=[0, 1], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay, lrConstant=0.09,
     hoodConstant=300, showint=0, show_sleep=1, final_sleep=20, network_dims=[3, 4])



"""
TODO:
- Visualize at step k (for ring)
- Neighborhood decay (exponentialatm)
- lrate decay (power atm)
- TOPOGRAPHY
x Normalize input
- Find path distance
- Create initial weight ring
- Implement for mnist
"""