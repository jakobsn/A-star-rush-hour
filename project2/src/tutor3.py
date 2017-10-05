import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT

# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman,lrate=.1,showint=None,mbs=10,vint=None,softmax=False):
        self.learning_rate = lrate
        self.layer_sizes = dims # Sizes of each layer of neurons
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 1000 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        print("************grabvar**************")
        print("modules length:", len(self.modules))
        print("modules:", self.modules)
        for module in self.modules:
            print("GANN MODULE:")
            print("in:", module.getvar("in"))
            print("out:", module.getvar("out"))
            print("biases:", module.getvar("bias"))
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())
        print("*********************************")

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
            gmod = Gannmodule(self,i,invar,insize,outsize)
            invar = gmod.output; insize = gmod.outsize
        self.output = gmod.output # Output of last module is output of whole network
        if self.softmax_outputs: self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        self.error = tf.reduce_mean(tf.square(self.target - self.output),name='MSE')
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

    def do_testing(self,sess,cases,msg='Testing'):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        error, grabvals, _ = self.run_one_step(self.error, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder,  show_interval=None)
        print('%s Set Error = %f ' % (msg, error))
        return error  # self.error uses MSE, so this is a per-case value

    def do_mapping(self):
        #TODO
        return

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing')

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation Testing')
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training')

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                  session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars,step=1):
        names = [x.name for x in grabbed_vars];
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        print("grabvals:", grabbed_vals)
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                print("plotting this matrix", names[i], ":")
                for row in v:
                    print(row)
                TFT.hinton_plot(v,fig=self.grabvar_figures[fig_index],title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            # Print graphical visialization of the bias vector in the module
            elif "bias" in names[i]:
                #TODO: Support long vectors
                v = np.array([v])
                print("v inn",v)
                TFT.display_matrix(v, title=names[i] + ' at step ' + str(step))
                fig_index += 1
            else:
                print(v, end="\n\n")


    def run(self,epochs=100,sess=None,continued=False):
        PLT.ion()
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session)
        self.testing_session(sess=self.current_session)
        self.close_current_session()
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True)

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

    def close_current_session(self):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=True)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self,ann,index,invariable,insize,outsize):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.build()

    def build(self):
        mona = self.name; n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize,n)),
                                   name=mona+'-wgt',trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector
        self.output = tf.nn.relu(tf.matmul(self.input,self.weights)+self.biases,name=mona+'-out')
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

    def __init__(self,cfunc,vfrac=0,tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
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
# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.

def autoex(epochs=100,nbits=3,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=True):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size, nbits, size],cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(0,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'out') # Add a grabvar (to be displayed in its own matplotlib window).
    #ann.add_grabvar(0,'bias') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    ann.runmore(epochs*2)
    return ann

def parity(epochs=100,nbits=2,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : TFT.gen_vector_count_cases(2, 2**nbits))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size, nbits, size+1],cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).

    ann.add_grabvar(0,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'out') # Add a grabvar (to be displayed in its own matplotlib window).

    ann.run(epochs)
    ann.runmore(epochs*2)
    return ann

def readFile(targetFile):
    with open(targetFile) as file:
        data = []
        for line in file:
            row = []
            elements = []
            for element in line.replace("\n", "").split(","):
                elements.append(float(element))
            row.append(elements)
            row.append([elements.pop()])
            data.append(row)
    return data

def datasets(epochs=100,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False,targetFile="../data/glass.txt"):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : readFile(targetFile))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size, nbits, size],cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    ann.runmore(epochs*2)
    return ann

def main(epochs=100, nbits=3, dims=[9, 3, 9], lrate=0.03, weight_range=None, showint=100, vint=100,
         data_params=(9,), data_funct=TFT.gen_all_one_hot_cases, steps=10, loss_funct=False,
         hl_activation_funct=False, op_activation_funct=True, case_fraction=1, vfrac=0.1, tfrac=0.1, mbs=8,
         map_batch_size=0, map_layers=0, map_dendrograms=[0], display_weights=[0], display_biases=[0]):
    #TODO: Find dims automaticly
    size = 2 ** nbits
    mbs = mbs if mbs else size
    case_generator = (lambda: data_funct(*data_params))
    cman = Caseman(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=dims,cman=cman,lrate=lrate,showint=showint,mbs=mbs,vint=vint,softmax=op_activation_funct)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(0,'bias') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(0,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'in') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.add_grabvar(1,'out') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs)
    #ann.runmore(epochs*2)
    return ann

#Translate default autoencoder to use generic main method
main(epochs=300,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,op_activation_funct=False, data_params=(2**4,), dims=[16, 4, 16])

#parity
#main(epochs=100, nbits=4, dims=[10, 2, 10+1], lrate=0.03, weight_range=None, showint=100, vint=100, data_params=(10, 10), data_funct=TFT.gen_vector_count_cases,
#         steps=10, loss_funct=False, hl_activation_funct=False, op_activation_funct=True, case_fraction=1, vfrac=0.1, tfrac=0.1, mbs=10,
#         map_batch_size=0, map_layers=0, map_dendrograms=[0], display_weights=[0], display_biases=[0])

#Default autoencoder
#autoex(epochs=300,nbits=4,lrate=0.03,showint=100,mbs=None,vfrac=0.1,tfrac=0.1,vint=100,sm=False)



"""
TODO:
- support activation_functs: hyperbolic tangent, sigmoid, relu or softmax
- hl_activation_funct: must be set in output < build < gannmodule
- op_activation_funct: must replace softmax parameter, and set in output < build < gann
- loss function: must be set in error < configure_learning < gann, either mean-squared error or cross entropy (mean-squared atm)
- initial weight range: must be set in weights < build < gannmodule
- datasource: specify function and param for case generator
- casefraction: length of sublist ca of cases < organize cases < caseman
- implement do_mapping, use ann.grabvar()
- how to  show graphical visualization of output layer
- support long bias vectors
- visualize dendrograms
- Find dims automaticly

Qs:
- Steps == global_training_step/epochs?
"""

#main()
#parity()
#autoex()
#Gann.reopen_current_session()
#data=readFile("../data/glass.txt")
#datasets()
#for line in data:
#    print(line)
