import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config

from theano_toolkit import utils as U
def build(P,input_size=None,layer_sizes=None,output_size=None):
	input_size = input_size or config.input_size
	layer_sizes = layer_sizes or config.layer_sizes
	output_size = output_size or config.output_size



