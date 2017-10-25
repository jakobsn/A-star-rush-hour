import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow import identity
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
from tflowtools import meanSquaredError, crossEntropy, readFile, scale_average_and_deviation, \
    scale_min_max, get_mnist_data, readShrooms, gen_segmented_vector_cases, gen_vector_count_cases, \
    gen_all_parity_cases, gen_all_one_hot_cases
from time import time, sleep
from math import ceil, floor
import argparse
import sys

relu = tf.nn.relu
softmax = tf.nn.softmax



# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman, weight_range=[-.1,.1], lrate=.1,showint=None,mbs=10,vint=None,ol_funct=tf.nn.softmax, hl_funct=tf.nn.relu, loss_funct=meanSquaredError):
        self.learning_rate = lrate
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.ol_funct = ol_funct
        self.hl_funct = hl_funct
        self.loss_funct = loss_funct
        self.modules = []
        self.weight_range = weight_range
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module): self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i,outsize in enumerate(self.layer_sizes[1:]):
            if i < (len(self.layer_sizes) - 2):
                gmod = Gannmodule(self,i,invar,insize,outsize, a_funct=self.hl_funct, weight_range=self.weight_range)
            else:
                gmod = Gannmodule(self,i,invar,insize,outsize, a_funct=self.ol_funct, weight_range=self.weight_range)
            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        self.error = self.loss_funct(self.target, self.output)
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error,name='Backprop')

    def do_training(self,sess,cases,epochs=100,continued=False):
        if not(continued): self.error_history = []
        for i in range(epochs):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step,show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self,sess,cases,msg='Testing',bestk=None):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def do_mapping(self, map_batch_size, map_layers, map_dendrograms, display_weights, display_biases,
                   onezero=False, komma=False, punktum=False, decimals=1, mapping_time=30, dendro_time=30):
        self.reopen_current_session()
        for layer in map_layers:
            self.add_grabvar(layer, 'in')  # Add a grabvar (to be displayed in its own matplotlib window).
            self.add_grabvar(layer, 'out')  # Add a grabvar (to be displayed in its own matplotlib window).
        for weight in display_weights:
            self.add_grabvar(weight, 'wgt')
        for bias in display_biases:
            self.add_grabvar(bias, "bias")

        inputs = [c[0] for c in self.caseman.get_training_cases()[:map_batch_size]]; targets = [c[1] for c in self.caseman.get_training_cases()[:map_batch_size]]
        print("input")
        print(inputs)
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=self.current_session, step="Postprocessing",
                                                 feed_dict=feeder, show_interval=None, mapping=True)
        # Close mapping session
        sleep(mapping_time)
        PLT.close("all")
        self.grabvars = []
        self.grabvar_figures = []

        # Start dendrogram session
        for dendro in map_dendrograms:
            self.add_grabvar(dendro, 'in')  # Add a grabvar (to be displayed in its own matplotlib window).
            self.add_grabvar(dendro, 'out')  # Add a grabvar (to be displayed in its own matplotlib window).
        try:
            testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=self.current_session, step="Postprocessing",
                                                     feed_dict=feeder, show_interval=None, mapping=False, dendrogram=True,
                                                     onezero=onezero, komma=komma, punktum=punktum, decimals=decimals, dendro_time=dendro_time)
        except:
            print("Input labels to large, shrinking...")
            try:
                if decimals > 1:
                    decimals -= 1
                elif punktum:
                    punktum = False
                elif komma:
                    komma = False
                testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=self.current_session, step="Postprocessing",
                                                         feed_dict=feeder, show_interval=None, mapping=False, dendrogram=True,
                                                         onezero=onezero, komma=komma, punktum=punktum, decimals=decimals, dendro_time=dendro_time)
            except:
                try:
                    if not onezero:
                        onezero = True
                    elif punktum:
                        punktum = False
                    elif komma:
                        komma = False
                    testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes,
                                                             session=self.current_session, step="Postprocessing",
                                                             feed_dict=feeder, show_interval=None, mapping=False, dendrogram=True,
                                                             onezero=onezero, komma=komma, punktum=punktum, decimals=decimals,dendro_time=dendro_time)
                except:
                    testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=self.current_session, step="Postprocessing",
                                                             feed_dict=feeder, show_interval=None, mapping=False, dendrogram=True,
                                                             onezero=True, komma=False, punktum=False, decimals=1,dendro_time=dendro_time)

        print('%s Set Error = %f ' % ("Map testing", testres))
        self.close_current_session()
        return

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess,bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',bestk=bestk)

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1, mapping=False, dendrogram=False,
                     onezero=False, komma=False, punktum=False, decimals=1, dendro_time=30):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        if mapping:
            self.display_grabvars(results[1], grabbed_vars, step=step)
        if dendrogram:
            self.display_dendrograms(results[1], grabbed_vars, step=step, onezero=onezero, komma=komma, punktum=punktum, decimals=decimals, sleep_time=dendro_time)
        return results[0], results[1], sess

    def display_dendrograms(self, grabbed_vals, grabbed_vars, step, onezero=False, komma=False, punktum=False, decimals=1, sleep_time=30):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        for j, i, o in zip(range(len(grabbed_vals[0:len(grabbed_vals):2])), grabbed_vals[0:len(grabbed_vals):2], grabbed_vals[1:len(grabbed_vals):2]):
            in_pattern=[]
            for line in i:
                in_vals = line
                for element in line:

                    # Format long floats, if one of the numbers is not an int.
                    if not element.is_integer():
                        in_vals = []
                        for element in line:
                            # Represent '0,0' as 'o'
                            if element == 0 and onezero:
                                in_vals.append("o")
                            # Ceil the float by with specified number of decimals
                            else:
                                deci = "%."+str(decimals)+"f"
                                in_vals.append(deci % element)
                        break

                # Keep '.'
                if punktum:
                    din = TFT.bits_to_str(in_vals, komma)
                # Remove '.'
                else:
                    din = TFT.bits_to_str(in_vals, komma).replace(".", "")
                # Gather only unique cases
                print("incoming din,", din)
                if not din in in_pattern:
                    in_pattern.append(din)
                if not self.allUnique(in_pattern):
                    in_pattern.pop()

            TFT.dendrogram(o, in_pattern, title="dendrogram: " + names[j], sleep_time=sleep_time)

    def allUnique(self, x):
        seen = set()
        return not any(i in seen or seen.add(i) for i in x)

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                #TFT.display_matrix(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            # Print graphical visialization of the bias vector in the module
            elif "bias" in names[i]:
                v = np.array([v])
                print("Biases",v)
                #TFT.display_matrix(v, title=names[i] + ' at step ' + str(step))
                TFT.hinton_plot(v, title="Biases: " + names[i])
                fig_index += 1
            else:
                print(v, end="\n\n")

    def run(self,epochs=100,sess=None,continued=False,bestk=None):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        self.testing_session(sess=self.current_session,bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize, a_funct, weight_range):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.a_funct = a_funct
        self.weight_range = weight_range
        self.build()

    def build(self):
        mona = self.name; n = self.outsize
        self.weights = tf.Variable(np.random.uniform(self.weight_range[0], self.weight_range[1], size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(self.weight_range[0], self.weight_range[1], size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        if self.a_funct:
            self.output = self.a_funct(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
        else:
            self.output = tf.matmul(self.input, self.weights)
        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0, cfrac=1):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.case_fraction = cfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases[:ceil((len(self.cases)*self.case_fraction))])
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


#   ****  MAIN functions ****

#main(data_funct=TFT.gen_vector_count_cases, data_params=(500, 15), epochs=100, dims=[15, 6, 16], hl_funct=tf.nn.relu, ol_funct=tf.nn.relu)

def main(data_funct=readFile, data_params=("../data/glass.txt","avgdev"), epochs=1000, dims=[9, 9, 7], lrate=0.1, mbs=10, showint=0, vint=100,
         vfrac=0.1, tfrac=0.1,hl_funct=tf.nn.sigmoid, ol_funct=tf.nn.softmax, loss_funct=crossEntropy, weight_range=[-.1, .1],
         cfrac=1, map_batch_size=0,map_layers=[], map_dendrograms=[], display_weights=[], display_biases=[], bestk=1,
         onezero=False, komma=True, punktum=False,  decimals=2, mapping_time=30, dendro_time=30):
    start = time()
    case_generator = (lambda : data_funct(*data_params))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac,cfrac=cfrac)
    ann = Gann(dims=dims,cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,ol_funct=ol_funct, hl_funct=hl_funct, loss_funct=loss_funct, weight_range=weight_range)

    ann.run(epochs,bestk=bestk)
    end = time()
    print("params", data_params, "epochs", epochs, "dims", dims, "lrate", lrate, "mbs", mbs)
    print("Time elapsed:", end - start, "s", (end-start)/60, "m")
    if map_batch_size:
        if not len(map_layers):
            ann.do_mapping(map_batch_size, map_layers, map_dendrograms, display_weights, display_biases, onezero, komma, punktum, decimals, mapping_time=0)
        else:
            ann.do_mapping(map_batch_size, map_layers, map_dendrograms, display_weights, display_biases, onezero, komma, punktum, decimals, mapping_time, dendro_time)
    #ann.runmore(1000,bestk=bestk)

    return ann


def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Takes input from command line
    parser = argparse.ArgumentParser(description='General Artificial Neural Network')
    parser.add_argument('-fu', type=str, help='Data function')
    parser.add_argument('-dp', type=str, help='Data parameters, separated with \",\"')
    parser.add_argument('-di', type=str,help='Dimensions, separated with \",\"')
    parser.add_argument('-ep', type=int,help='Epochs',nargs='?')
    parser.add_argument('-hl', type=str, help='Hidden layer activation function', nargs = '?')
    parser.add_argument('-ol', type=str,help='Output layer activation function',nargs='?')
    parser.add_argument('-lf', type=str,help='Loss function, \'crossEntropy\' or \'meanSquaredError\'',nargs='?')
    parser.add_argument('-lr', type=float,help='Learning rate, default is 0.3',nargs='?')
    parser.add_argument('-mbs', type=int,help='Mini batch size, default is 10',nargs='?')
    parser.add_argument('-ms', type=int,help='Map batch size',nargs='?')
    parser.add_argument('-ml', type=str, help='Map layers separated with \",\"', nargs='?')
    parser.add_argument('-md', type=str, help='Map dendrogerams layers separated with \",\"', nargs='?')
    parser.add_argument('-db', type=str, help='Display biases separated with \",\"', nargs='?')
    parser.add_argument('-dw', type=str, help='Display weights separated with \",\"', nargs='?')
    parser.add_argument('-bk', type=int, help='Best k', nargs='?')
    parser.add_argument('-oz', type=str2bool, help='Display \"0.0\" as \"o\" in dendrograms?', nargs='?')
    parser.add_argument('-ko', type=str2bool, help='Include \',\' in dendrograms?', nargs='?')
    parser.add_argument('-pu', type=str2bool, help='Include \'.\' in dendrograms?', nargs='?')
    parser.add_argument('-de', type=int, help='Decimals included in each feature in dendrograms', nargs='?')
    parser.add_argument('-mt', type=int, help='Mapping time', nargs='?')
    parser.add_argument('-dt', type=int, help='Dendrogram time', nargs='?')
    parser.add_argument('-si', type=int, help='Show interval', nargs='?')
    parser.add_argument('-vi', type=int, help='Validation interval', nargs='?')

    args = parser.parse_args()

    funct = globals()[args.fu]
    parameters = args.dp.split(',')
    # If the function is not a file reader, the arguments must be converted to int
    if funct is not get_mnist_data and funct is not readFile and funct is not readShrooms:
        for i, j in enumerate(parameters):
            parameters[i] = int(j)

    dimensions = args.di.split(',')
    for i, j in enumerate(dimensions):
        dimensions[i] = int(j)

    if args.hl: hl_funct = globals()[args.hl]
    if args.ol: ol_funct = globals()[args.ol]
    if args.lf: loss_funct = globals()[args.lf]

    if args.ml:
        map_layers = args.ml.split(',')
        for i, j in enumerate(map_layers):
            map_layers[i] = int(j)
    else:
        map_layers = []

    if args.md:
        map_dendrograms = args.md.split(',')
        for i, j in enumerate(map_dendrograms):
            map_dendrograms[i] = int(j)
    else:
        map_dendrograms = []

    if args.db:
        display_biases = args.ml.split(',')
        for i, j in enumerate(display_biases):
            display_biases[i] = int(j)
    else:
        display_biases = []

    if args.dw:
        display_weights = args.ml.split(',')
        for i, j in enumerate(display_weights):
            display_weights[i] = int(j)
    else:
        display_weights = []


    if not args.hl: hl_funct = relu
    if not args.ol: ol_funct = softmax
    if not args.lf: loss_funct = crossEntropy
    if not args.lr: args.lr = 0.3
    if not args.mbs: args.mbs = 10
    if not args.bk: args.bk = 1
    if not args.oz: args.oz = False
    if not args.ko: args.ko = True
    if not args.pu: args.pu = True
    if not args.de: args.de = 2
    if not args.mt: args.mt = 20
    if not args.dt: args.dt = 20
    if not args.si: args.si = 0
    if not args.vi: args.vi = 500

    main(data_funct=funct, data_params=parameters,dims=dimensions, hl_funct=hl_funct, ol_funct=ol_funct,
         loss_funct=loss_funct, lrate=args.lr, mbs=args.mbs, map_layers=map_layers, map_dendrograms=map_dendrograms,
         display_biases=display_biases, display_weights=display_weights, bestk=args.bk, onezero=args.oz, komma=args.ko,
         punktum=args.pu, decimals=args.de, mapping_time=args.mt, dendro_time=args.mt, showint=args.si, vint=args.vi)



"""
main(data_funct=TFT.gen_all_parity_cases, data_params=(4,), epochs=100, dims=[4, 20, 2], lrate=0.2, mbs=30,
         hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=crossEntropy, map_batch_size=30, map_layers=[],
         display_biases=[], map_dendrograms=[0,1], onezero=False, komma=True, punktum=True, decimals=2); print("relu, tan, ce")
"""




#autoencoder, 100%. WILL NOT BE TESTED using gen_dense_autoencoder_cases is an option
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=2000,nbits=4, dims=[2**4, 4, 2**4],lrate=0.1,showint=10000,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,vint=10000,ol_funct=tf.nn.relu, hl_funct=tf.nn.relu,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1)
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.nn.tanh, hl_funct=tf.nn.relu,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1); print("tan, relu, ce")
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.identity, hl_funct=tf.nn.relu,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1); print("id, relu, ce")
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.nn.softmax, hl_funct=tf.nn.sigmoid,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1); print("sm, sig, ce")
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.nn.tanh, hl_funct=tf.nn.relu,loss_funct=meanSquaredError,weight_range=[0, 1],bestk=1); print("tan, relu, mse")

#~30%
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.identity, hl_funct=tf.nn.tanh,loss_funct=meanSquaredError,weight_range=[0, 1],bestk=1); print("id, tan, mse")
#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=1000, dims=[2**4, 4, 2**4],lrate=0.1,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,ol_funct=tf.identity, hl_funct=tf.nn.sigmoid,loss_funct=meanSquaredError,weight_range=[0, 1],bestk=1); print("sm, sig, ce")


# dataset wine,  100% DONE
#main(data_funct=readFile, data_params=("../data/wine.txt","avgdev", True), epochs=2000, dims=[11, 50, 20, 8], mbs=10, lrate=0.4, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError); print("tan, id, mse")

# dataset glass, 97-100% DONE
#2main(data_funct=readFile, data_params=("../data/glass.txt","avgdev"), epochs=1000, dims=[9, 10, 10, 7], mbs=10, lrate=0.5, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError); print("tan, id, mse")
#main(data_funct=readFile, data_params=("../data/glass.txt","avgdev"), epochs=1000, dims=[9, 10, 10, 7], mbs=10, lrate=0.7, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError); print("tan, id, mse")
#main(data_funct=readFile, data_params=("../data/glass.txt","avgdev"), epochs=1000, dims=[9, 10, 10, 7], mbs=10, lrate=0.7, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError, map_batch_size=5, map_dendrograms=[0,1,2]); print("tan, id, mse")


# dataset yeast, 94-100% DONE
#2main(data_funct=readFile, data_params=("../data/yeast.txt","avgdev"), epochs=500, dims=[8, 70, 50, 10], mbs=5, lrate=0.3, hl_funct=tf.nn.relu, ol_funct=tf.identity, loss_funct=crossEntropy); print("relu, id, ce")
#main(data_funct=readFile, data_params=("../data/yeast.txt","avgdev"), epochs=500, dims=[8, 60, 50, 10], mbs=5, lrate=0.5, hl_funct=tf.nn.relu, ol_funct=tf.identity, loss_funct=crossEntropy, map_batch_size=5, map_dendrograms=[0,1,2]); print("relu, id, ce")

#parity, 95-100% DONE
#2main(data_funct=TFT.gen_all_parity_cases, data_params=(10,), epochs=1000, dims=[10, 50, 2], lrate=0.2, mbs=20, hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=crossEntropy); print("relu, tan, ce")
#main(data_funct=TFT.gen_all_parity_cases, data_params=(10,), epochs=1000, dims=[10, 50, 2], lrate=0.2, mbs=30, hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=crossEntropy, map_batch_size=5, map_layers=[0], display_biases=[0]); print("relu, tan, ce")

#countex, 97-100% DONE
#main(data_funct=TFT.gen_vector_count_cases, data_params=(500, 15), epochs=4000, dims=[15, 55, 20, 16], hl_funct=tf.nn.sigmoid, ol_funct=tf.identity, loss_funct=meanSquaredError, lrate=0.6, mbs=5)#, map_batch_size=10, map_layers=[0,1])

#segment counter 99% DONE
#main(data_funct=TFT.gen_segmented_vector_cases, data_params=(25, 1000, 0, 8), epochs=1000, dims=[25, 30, 10, 9], lrate=0.6,mbs=20,vfrac=0.1,tfrac=0.1,cfrac=1, ol_funct=tf.identity , hl_funct=tf.nn.tanh, loss_funct=meanSquaredError, bestk=1)#, map_batch_size=10, map_layers=[0,2])

# dataset, mushrooms, 95-96%. DONE. Classifies mushrooms from agaricus and lepiota family as poisonous or edible. https://archive.ics.uci.edu/ml/datasets/Mushroom
#main(data_funct=readShrooms, data_params=("../data/agaricus-lepiota.data",), epochs=100, dims=[22, 2], mbs=10, hl_funct=tf.nn.sigmoid, ol_funct=tf.nn.softmax, loss_funct=crossEntropy, map_batch_size=10, map_layers=[0], map_dendrograms=[0], display_weights=[0], display_biases=[0])

# MNIST DONE
#main(data_funct=get_mnist_data, data_params=(17230,), epochs=100, dims=[784, 600, 10], lrate=0.2, mbs=200, hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=meanSquaredError ,cfrac=0.1,map_batch_size=5, map_layers=[0, 1], map_dendrograms=[0, 1], display_weights=[], display_biases=[])


#main(data_funct=get_mnist_data, data_params=(17230,), epochs=100, dims=[784, 600, 10], lrate=0.2, mbs=200, hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=meanSquaredError ,cfrac=0.1,map_batch_size=5, map_dendrograms=[0, 1], display_weights=[], display_biases=[])
#main(data_funct=readShrooms, data_params=("../data/agaricus-lepiota.data",), epochs=100, dims=[22, 2], mbs=10, hl_funct=tf.nn.sigmoid, ol_funct=tf.nn.softmax, loss_funct=crossEntropy, map_batch_size=10, map_layers=[], map_dendrograms=[0], display_weights=[], display_biases=[0])
#main(data_funct=TFT.gen_segmented_vector_cases, data_params=(25, 1000, 0, 8), epochs=1000, dims=[25, 30, 10, 9], lrate=0.6,mbs=20,vfrac=0.1,tfrac=0.1,cfrac=1, ol_funct=tf.identity , hl_funct=tf.nn.tanh, loss_funct=meanSquaredError, map_batch_size=5, map_dendrograms=[0, 1, 2], bestk=1)#, map_batch_size=10, map_layers=[0,2])
#main(data_funct=TFT.gen_vector_count_cases, data_params=(500, 15), epochs=4000, dims=[15, 55, 20, 16], hl_funct=tf.nn.sigmoid, ol_funct=tf.identity, loss_funct=meanSquaredError, lrate=0.6, mbs=5 ,map_batch_size=5, map_dendrograms=[0, 1, 2])#, map_batch_size=10, map_layers=[])
#main(data_funct=TFT.gen_all_parity_cases, data_params=(10,), epochs=1000, dims=[10, 50, 2], lrate=0.2, mbs=30, hl_funct=tf.nn.relu, ol_funct=tf.nn.tanh, loss_funct=crossEntropy, map_batch_size=5, map_layers=[], map_dendrograms=[0, 1], display_biases=[]); print("relu, tan, ce")
#main(data_funct=readFile, data_params=("../data/yeast.txt","avgdev"), epochs=500, dims=[8, 60, 50, 10], mbs=5, lrate=0.5, hl_funct=tf.nn.relu, ol_funct=tf.identity, loss_funct=crossEntropy, map_batch_size=5, map_dendrograms=[0, 1, 2]); print("relu, id, ce")
#main(data_funct=readFile, data_params=("../data/glass.txt","avgdev"), epochs=1000, dims=[9, 10, 10, 7], mbs=10, lrate=0.7, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError, map_batch_size=5, map_dendrograms=[0, 1, 2]); print("tan, id, mse")
#main(data_funct=readFile, data_params=("../data/wine.txt","avgdev", True), epochs=2000, dims=[11, 50, 20, 8], mbs=10, lrate=0.4, hl_funct=tf.nn.tanh, ol_funct=tf.identity, loss_funct=meanSquaredError, map_batch_size=5, map_dendrograms=[0, 1, 2]); print("tan, id, mse")

"""
TODO:
x support activation_functs: hyperbolic tangent, sigmoid, relu or softmax
x hl_activation_funct: must be set in output < build < gannmodule
x op_activation_funct: must replace softmax parameter, and set in output < build < gann
x loss function: must be set in error < configure_learning < gann, either mean-squared error or cross entropy (mean-squared atm)
x initial weight range: must be set in weights < build < gannmodule
x datasource: specify function and param for case generator
x casefraction: length of sublist ca of cases < organize cases < caseman
x implement do_mapping, use ann.grabvar()
x how to  show graphical visualization of output layer
x support long bias vectors (maximize window...)
x visualize dendrograms
x Find dims automaticly (or not?) not
x Implement reading from mnist
x Implement scaling by deviation
x Numpy arrays

Qs:
- Steps == global_training_step/epochs?
"""

# IMPORTANT TFLOW FUNCTIONS:
# tf.nn.sigmoid
# tf.nn.relu
# tf.nn.tanh
# identity


#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=2000,nbits=4, dims=[2**4, 4, 2**4],lrate=0.1,showint=10000,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,vint=10000,ol_funct=tf.nn.relu, hl_funct=tf.nn.relu,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1)


#main(data_funct=TFT.gen_all_one_hot_cases, data_params=(2**4,), epochs=10000,nbits=4, dims=[2**4, 4, 2**4],lrate=0.1,showint=10000,mbs=10,vfrac=0.1,tfrac=0.1, cfrac=1,vint=10000,ol_funct=tf.nn.relu, hl_funct=tf.nn.relu,loss_funct=crossEntropy,weight_range=[0, 1],bestk=1)


# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(epochs=10000,nbits=4,lrate=0.1,showint=1000,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False,bestk=1):
    size = 2**nbits
    mbs = 10
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size,nbits,size], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, ol_funct=tf.nn.softmax, hl_funct=tf.nn.relu,loss_funct=crossEntropy)

    #ann.add_grabvar(0, 'in')  # Add a grabvar (to be displayed in its own matplotlib window).
    #ann.add_grabvar(1, 'out')  # Add a grabvar (to be displayed in its own matplotlib window).    ann.run(epochs,bestk=bestk)
    ann.run(epochs, bestk=bestk)
    sleep(10)
    #ann.runmore(epochs,bestk=bestk)
    return ann

def countex(epochs=10000,nbits=10,ncases=500,lrate=0.1,showint=1000,mbs=20,vfrac=0.1,tfrac=0.1,vint=200,sm=True,bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[nbits, nbits*3, nbits+1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint)
    #ann.add_grabvar(0, 'in')  # Add a grabvar (to be displayed in its own matplotlib window).
    #ann.add_grabvar(1, 'out')  # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs,bestk=bestk)
    return ann
#countex()
#autoex()

#main(data_funct=TFT.gen_vector_count_cases, data_params=(500, 15), epochs=3000, dims=[15, 6, 16], mbs=20,
#     hl_funct=tf.nn.relu, ol_funct=tf.nn.softmax, loss_funct=meanSquaredError, showint=0, map_batch_size=5, map_layers=[0, 1], map_dendrograms=[0, 1], display_weights=[], display_biases=[])

#main(data_funct=TFT.gen_vector_count_cases, data_params=(500, 15), epochs=100, dims=[15, 6, 16], hl_funct=tf.nn.relu, ol_funct=tf.nn.relu,map_batch_size=5, map_layers=[0, 1], map_dendrograms=[0, 1], display_weights=[], display_biases=[])

# Cross entropy + softmax + one hot