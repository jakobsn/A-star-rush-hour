
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
from tutor3 import *

# should epochs be a parameter?
def main(dims, hl_activation_funct, op_activation_funct, loss_funct, \
         lrate, weight_range, data_params, data_funct, case_fraction=1, \
         validation_fraction=1, test_fraction=1, minibatch_size=10, \
         map_batch_size=0, steps=10, map_layers=0, map_dendrograms=[0], \
         display_weights=[0], display_biases=[0]):

    autoex()
    return


def logistic(input):
    return 1.0/(1.0+np.exp(-input))


if __name__ == '__main__':
    main([4, 16, 4], )
