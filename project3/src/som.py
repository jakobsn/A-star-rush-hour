#!/usr/bin/python

import numpy as np
import somtools as st
from somtools import exponentialDecay, powerDecay, readTSP, get_mnist_data
from math import sqrt, ceil, floor
import matplotlib.pyplot as PLT
from matplotlib import colors, colorbar
from time import sleep, time
from random import randint, sample
import argparse

class SOM:
    def __init__(self, epochs, lrate, hoodsize, insize, outsize, features,
                 weight_range, lrate_decay, hood_decay, lrConstant,
                 hoodConstant, showint,show_sleep, network_dims, sort, radius,
                 problem, full_screen, mbs, vsize):
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
        self.radius = radius
        self.problem = problem
        self.full_screen = full_screen
        self.mbs = mbs
        if insize is None or outsize is None:
            self.insize = len(features[0])
            self.outsize = len(features)*2
        else:
            self.insize = insize
            self.outsize = outsize
        if self.network_dims is not None:
            self.topo = "matrix"
            self.neuron_matrix = self.create_neuron_matrix()
            self.triggered_targets = [None] * self.outsize
            for targets in range(len(self.triggered_targets)):
                self.triggered_targets[targets] = list()
            self.weights = self.initial_weights(sort)
        else:
            self.topo = "ring"
            self.triggered_targets = None
            self.weights = st.generate_points(self.weight_range[0], self.weight_range[1], self.radius, 0.1, self.outsize)
        self.neuronRing = self.create_neuron_ring()
        self.input_vector = None # Keep track of current input vector
        self.i = 0 # Keep track of which weight being changed for input neuron
        self.isDisplayed = False
        self.showint = showint
        self.show_sleep = show_sleep
        self.vsize = vsize
        if self.mbs < 1:
            self.do_training()
        else:
            self.do_batch_training()

    def do_batch_training(self):
        self.neuronRing = self.create_neuron_ring()
        for epoch in range(self.epochs):
            if self.showint != 0 and (epoch % self.showint == 0):
                print(str('[' + str(epoch) + ']'), "hood, lrate", self.hoodsize, self.lrate)
                if self.topo == "ring":
                    distance, path = self.findTotalDistance()
                    self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep, path, distance)
                else:
                    self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep)

            if self.lrate == 0:
                break
            if self.topo == "matrix":
                features = self.features
                np.random.shuffle(features)
                features = []
                targets = []
                sets = sample(range(0, len(self.features)), self.mbs)
                for s in sets:
                    features.append(self.features[s][0])
                    targets.append(self.features[s][1])
            else:
                features = self.features
                np.random.shuffle(features)
                features = features[:self.mbs]

            winners = []
            for feature in features:
                winners.append(self.findWinner(feature))

            for i in range(len(winners)):
                neighbours = self.get_neighbours(winners[i])
                self.input_vector = features[i]
                self.adjust_clusters(neighbours)
                if self.topo == "matrix":
                    self.triggered_targets[winners[i]].append(targets[i])
            self.hoodsize = round(self.hood_decay(epoch, self.initial_hood, self.hoodConstant, self.epochs))
            self.lrate = self.lrate_decay(epoch, self.initial_lrate, self.lrConstant, self.epochs)

        if self.topo == "ring":
            distance, path = self.findTotalDistance()
            if self.show_sleep:
                self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep, path, distance)
                sleep(self.show_sleep)
        else:
            print("test train")
            self.do_testing(self.features)
            print("test test")
            self.do_testing(st.get_mnist_test_data(self.vsize))
            if self.show_sleep:
                self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep)
                sleep(self.show_sleep)

    # Pick random input feature
    # Update weights for winner and neighbours
    # Update lrate and hoodsize
    def do_training(self):
        self.neuronRing = self.create_neuron_ring()
        for epoch in range(self.epochs):

            if self.showint != 0 and (epoch % self.showint == 0):
                print(str('[' + str(epoch) + ']'), "hood, lrate", self.hoodsize, self.lrate)
                if self.topo == "ring":
                    distance, path = self.findTotalDistance()
                    self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep, path, distance)
                else:
                    self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep)
                    print("Neuron ring", self.neuronRing)

            if self.lrate == 0:
                break
            if self.topo == "matrix":
                feature = self.features[randint(0, len(self.features)-1)][0]
                target = self.features[randint(0, len(self.features)-1)][1]
            else:
                feature = self.features[randint(0, len(self.features)-1)]
            #print("feature", feature)
            winner_neuron = self.findWinner(feature)
            neighbours = self.get_neighbours(winner_neuron)
            self.neuronRing[winner_neuron] += 1
            self.input_vector = feature
            self.adjust_clusters(neighbours)
            self.hoodsize = round(self.hood_decay(epoch, self.initial_hood, self.hoodConstant, self.epochs))
            self.lrate = self.lrate_decay(epoch, self.initial_lrate,self.lrConstant, self.epochs)
            if self.topo == "matrix":
                self.triggered_targets[winner_neuron].append(target)
            #print(str('[' + str(epoch) + ']'), "hood, lrate", self.hoodsize, self.lrate)
        if self.topo == "ring":
            distance, path = self.findTotalDistance()
            if self.show_sleep:
                self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep, path, distance)
                sleep(self.show_sleep)
        else:
            print("test train")
            self.do_testing(self.features)
            print("test test")
            self.do_testing(st.get_mnist_test_data(self.vsize))
            if self.show_sleep:
                self.do_mapping(self.weight_range, self.hoodsize, self.lrate, self.epochs, epoch, self.show_sleep)
                sleep(self.show_sleep)

    def do_testing(self, datasets):
        correct = 0
        cases = 0
        axises = np.arange(self.outsize)
        average_targets = []
        for targets in self.triggered_targets:
            if len(targets):
                average_target = np.mean(targets, axis=axises[0])
                target_number = np.argmax(average_target)
                average_targets.append(target_number)
            else:
                average_targets.append(None)
        for feature, target in datasets:
            winner_neuron = self.findWinner(feature)
            if average_targets[winner_neuron] == np.argmax(target):
                correct += 1
            cases += 1
        print("Correct:", correct, "of", cases)
        print("Precent;", 100*(correct/cases), "%")
        return

    # Find distance for TSP solution
    def findTotalDistance(self):
        actual_distance = 0
        path = self.findPath()
        for p in range(len(path[0])-1):
            node1, node2 = [path[0][p], path[1][p]], [path[0][p+1], path[1][p+1]]
            actual_distance += sqrt((abs(node1[0] - node2[0])**2 + abs(node1[1] - node2[1])**2))
        node1, node2 = [path[0][-1], path[1][-1]], [path[0][0], path[1][0]]
        actual_distance += sqrt((abs(node1[0] - node2[0])**2 + abs(node1[1] - node2[1])**2))
        print("********DISTANCE*************")
        print(actual_distance)
        print("*****************************")
        return actual_distance, path

    def findPath(self):
        path = [None] * len(self.weights[0])
        for nodes in range(len(self.weights[0])):
            path[nodes] = list()

        for feature in self.features:
            winner_neuron = self.findWinner(feature)
            path[winner_neuron].append(feature)
        flat_path = [[], []]
        for nodes in path:
            if len(nodes) == 0:
                pass
            elif len(nodes) == 1:
                flat_path[0].append(nodes[0][0])
                flat_path[1].append(nodes[0][1])
            else:
                for node in nodes:
                    flat_path[0].append(node[0])
                    flat_path[1].append(node[1])
                # TODO: Pick best sequence
        return flat_path

    def findWinner(self, feature):
        weights = np.rot90(self.weights, 1)[::-1]
        distances = self.euclidian_distance(weights, feature)
        winner_neuron = np.argmin(distances)
        return winner_neuron

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
                            if sum(hoodsizes) <= self.hoodsize:
                                matrix_neighbours[0].append(int(self.neuron_matrix[i][j]))
                                matrix_neighbours[1].append(sum(hoodsizes))
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

    def euclidian_distance(self, weights, input_vector):
        distances = []
        for weight_vector in weights:
            distances.append(np.sum(np.power(np.subtract(input_vector, weight_vector), 2)))
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
    def initial_weights(self, sort=False):
        weights = np.random.uniform(self.weight_range[0], self.weight_range[1], [self.insize, self.outsize])
        if sort:
            np.sort(weights)
        return weights

    def do_mapping(self, weight_range=None, hood=None, lrate=None, epochs=None, step='NA', sleep_time=1, path=None, distance=0):
        if hood == None: hood=self.hoodsize
        if epochs == None: epochs=self.epochs
        if weight_range == None: weight_range=self.weight_range
        if PLT.get_fignums():
            sleep(sleep_time)
            PLT.close("all")

        if self.full_screen:
            mng = PLT.get_current_fig_manager()
            mng.full_screen_toggle()

        if self.topo == "ring":
            self.map_tsp()
        else:
            self.map_mnist()

        if path != None:
            self.map_path(path)
            PLT.suptitle("Run: " + str(step) + " Distance: " + str(round(distance, 0)) + \
                         " Problem: " + str(self.problem).replace("../data/", "").replace("txt",""))
        else:
            PLT.suptitle("Run: " + str(step))
        PLT.title(" Epochs " + str(epochs) + " Lrate " + str(self.lrate) \
                     + " Hood " + str(hood) + " Weight range " + str(weight_range) + \
                     " Outsize " + str(self.outsize) + " constants " + str(self.lrConstant) + ',' + str(self.hoodConstant))
        PLT.show(block=False)
        if self.topo == 'matrix':
            self.mnist_show_colors()

    def map_path(self, path):
        PLT.plot(path[0], path[1], c='pink')
        PLT.plot([path[0][0], path[0][-1]], [path[1][0], path[1][-1]], c='pink')

    def map_mnist(self):
        axises = np.arange(self.outsize)
        coordinates = []
        color_list = ['red', 'blue', 'green', 'black', 'purple', 'yellow', 'pink', 'gray', 'brown', 'orange']
        for y in range(self.network_dims[0]):
            for x in range(self.network_dims[1]):
                coordinates.append([x, y])
        for targets in self.triggered_targets:
            x, y = coordinates.pop(0)
            if len(targets):
                average_target = np.mean(targets, axis=axises[0])
                target_number = np.argmax(average_target)
                PLT.scatter(x, y, c=color_list[target_number])
            else:
                PLT.scatter(x, y, c='white')

    def mnist_show_colors(self):
        color_list = ['red', 'blue', 'green', 'black', 'purple', 'yellow', 'pink', 'gray', 'brown', 'orange']
        fig, ax = PLT.subplots()
        cmap = colors.ListedColormap(color_list)
        cmap.set_over('0.25')
        cmap.set_under('0.75')
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        cb2 = colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        boundaries=[-1] + bounds + [11],
                                        extend='both',
                                        orientation='horizontal')
        cb2.set_label('Colors in range 0-9')
        fig.show()

    def map_tsp(self):
        # Scatter weights
        for wx, wy in zip(self.weights[0], self.weights[1]):
            PLT.scatter(wx, wy, c="red")
        # Plot weight edges
        PLT.plot(self.weights[0], self.weights[1])
        PLT.plot([self.weights[0][0], self.weights[0][-1]], [self.weights[1][0], self.weights[1][-1]], c='blue')
        # Scatter features
        for feature in self.features:
            PLT.scatter(feature[0], feature[1], c="black")


def main(data_funct=st.readTSP, data_params='../data/6.txt', epochs=4000,  lrate=0.1, hoodsize=6,
         insize=2, outsize=220, weight_range=[30, 30], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
         lrConstant=0.5, hoodConstant=3300, showint=0, show_sleep=10, network_dims=None,
         sort=False, radius=1, full_screen=False, mbs=0, vsize=100):
    if type(data_params) == int:
        features = data_funct(data_params)
    elif not data_params.isdigit():
        features = data_funct(data_params)
    else:
        features = data_funct(int(data_params))
    start = time()

    som = SOM(epochs=epochs, lrate=lrate, hoodsize=hoodsize, features=features, insize=insize, outsize=outsize,
              weight_range=weight_range, lrate_decay=lrate_decay, hood_decay=hood_decay, lrConstant=lrConstant,
              hoodConstant=hoodConstant, showint=showint, show_sleep=show_sleep, network_dims=network_dims,
              sort=sort, radius=radius, problem=data_params, full_screen=full_screen, mbs=mbs, vsize=vsize)
    print('funct', data_funct, 'params', data_params, 'epochs', epochs, '\n',
          'lrate', lrate, 'hoodsize', hoodsize, 'insize', insize, 'outsize', outsize,  'weight_range', weight_range, '\n',
          'lrate_deacay', lrate_decay, 'hood_decay', hood_decay,   '\n',
          'topo', som.topo, "lrConstant", lrConstant, "hoodConstant", hoodConstant, "mbs", mbs)
    end = time()
    print("Time elapsed:", end - start, "s", (end-start)/60, "m")


def str2funct(s):
    return globals()[s]


def str2list(s):
    li = s.split(',')
    for i, j in enumerate(li):
        li[i] = int(j)
    return li


if __name__ == '__main__':
    # Takes input from command line
    parser = argparse.ArgumentParser(description='General Artificial Neural Network')
    parser.add_argument('-fu', type=str2funct, help='Data function')
    parser.add_argument('-dp', type=str, help='Data parameter')
    parser.add_argument('-ep', type=int, help='Epochs', nargs='?')
    parser.add_argument('-di', type=str2list, help='Dimensions, separated with \",\"', nargs='?')
    parser.add_argument('-lr', type=float, help='Learning rate',nargs='?')
    parser.add_argument('-lc', type=float, help='Learning rate decay constant', nargs = '?')
    parser.add_argument('-lf', type=str2funct, help='Learning rate decay function', nargs = '?')
    parser.add_argument('-iz', type=int, help='Input size',nargs='?')
    parser.add_argument('-os', type=int, help='Output size',nargs='?')
    parser.add_argument('-hs', type=int, help='Hoodsize',nargs='?')
    parser.add_argument('-hc', type=float ,help='Hood constant',nargs='?')
    parser.add_argument('-hf', type=str2funct, help='Hood decay function', nargs='?')
    parser.add_argument('-wr', type=str2list, help='Weight range, separated with \",\"', nargs='?')
    parser.add_argument('-ss', type=int, help='Sleep on show', nargs='?')
    parser.add_argument('-si', type=int, help='Show interval', nargs='?')
    parser.add_argument('-mbs', type=int, help='Mini batch size',nargs='?')
    parser.add_argument('-ra', type=int, help='Radius',nargs='?')
    parser.add_argument('-vs', type=int, help='Number of validation cases',nargs='?')
    args = parser.parse_args()

    if args.ep is None: args.ep = 5000
    if args.di is None: args.di = None
    if args.lr is None: args.lr = 0.1
    if args.lc is None: args.lc = 0.5
    if args.lf is None: args.lf = powerDecay
    if args.iz is None: args.iz = 2
    if args.os is None: args.os = 220
    if args.hs is None: args.hs = 6
    if args.hc is None: args.hc = 3300
    if args.hf is None: args.hf = exponentialDecay
    if args.wr is None: args.wr = [50, 50]
    if args.ss is None: args.ss = 0
    if args.si is None: args.si = 0
    if args.mbs is None: args.mbs = 0
    if args.ra is None: args.ra = None
    if args.vs is None: args.vs = 100

    main(data_funct=args.fu, data_params=args.dp, epochs=args.ep, network_dims=args.di,
         lrate=args.lr, lrConstant=args.lc, lrate_decay=args.lf, insize=args.iz, outsize=args.os,
         hoodsize=args.hs, hoodConstant=args.hc, hood_decay=args.hf, weight_range=args.wr,
         show_sleep=args.ss, showint=args.si, mbs=args.mbs, radius=args.ra, vsize=args.vs)


"""
TODO:
x Batch training
x Visualize at step k (for ring)
x Decay (exponential and power atm)
x TOPOGRAPHY
x Normalize input
x Find path distance
x Create initial weight ring
x Visualize for mnist
x Check how well mnist is classified
"""

# SOLUTIONS

# Works well enough on mnist
# main(data_funct=st.get_mnist_data, data_params=900, epochs=9000, lrate=0.3, hoodsize=2, insize=784, outsize=100,
#     weight_range=[0, 1], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay, lrConstant=0.08,
#     hoodConstant=440, showint=0, show_sleep=10, network_dims=[10, 10], mbs=5)

# Mostly is within 10% on 1
# main(data_funct=st.readTSP, data_params='../data/1.txt', epochs=10000, lrate=0.1, hoodsize=6,
#        insize=2, outsize=150, weight_range=[800, 800], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=0.5, hoodConstant=3000, showint=0, show_sleep=5, network_dims=None,
#         sort=False, radius=1)

# Mostly is within 10% on 2
# main(data_funct=st.readTSP, data_params='../data/2.txt', epochs=11000, lrate=0.1, hoodsize=7,
#         insize=2, outsize=220, weight_range=[30, 30], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=1, hoodConstant=2000, showint=0, show_sleep=50, network_dims=None,
#         sort=False, radius=1, mbs=0)

# Good on 3
# main(data_funct=st.readTSP, data_params='../data/3.txt', epochs=8000, lrate=0.1, hoodsize=6,
#         insize=2, outsize=220, weight_range=[30, 30], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=1.3, hoodConstant=3300, showint=0, show_sleep=50, network_dims=None,
#         sort=False, radius=1, mbs=0)

# Good on 4
# main(data_funct=st.readTSP, data_params='../data/4.txt', epochs=9000, lrate=0.1, hoodsize=6,
#         insize=2, outsize=220, weight_range=[30, 30], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=1.3, hoodConstant=3300, showint=0, show_sleep=50, network_dims=None,
#         sort=False, radius=1, mbs=0)

# Mostly is within 10% on 5
# main(data_funct=st.readTSP, data_params='../data/5.txt', epochs=9000, lrate=0.08, hoodsize=6,
#         insize=2, outsize=200, weight_range=[30, 30], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=0.7, hoodConstant=3000, showint=0, show_sleep=10, network_dims=None,
#         sort=False, radius=1, mbs=0)

# Mostly is within 10% on 6
# main(data_funct=st.readTSP, data_params='../data/6.txt', epochs=10000, lrate=0.06, hoodsize=6,
#         insize=2, outsize=200, weight_range=[50, 50], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=0.9, hoodConstant=3500, showint=0, show_sleep=10, network_dims=None,
#         sort=False, radius=1, mbs=0)

# Mostly is within 10% on 7
# main(data_funct=st.readTSP, data_params='../data/7.txt', epochs=8001, lrate=0.1, hoodsize=6,
#         insize=2, outsize=250, weight_range=[9000, 6000], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=0.5, hoodConstant=3300, showint=200, show_sleep=2, network_dims=None,
#         sort=False, radius=10, mbs=0)

# Mostly is within 10% on 8
# main(data_funct=st.readTSP, data_params='../data/8.txt', epochs=10000, lrate=0.06, hoodsize=6,
#         insize=2, outsize=220, weight_range=[60, 110], lrate_decay=st.powerDecay, hood_decay=st.exponentialDecay,
#         lrConstant=0.7, hoodConstant=3300, showint=0, show_sleep=10, network_dims=None,
#         sort=False, radius=2, mbs=0)

