import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
from itertools import izip

def softmax(X):
    exp_X_ = T.exp(X - T.max(X,axis=-1,keepdims=True))
    return exp_X_ / T.sum(exp_X_,axis=-1,keepdims=True)

def log_sigmoid(x):
    x = x[:,0]
    return (-T.nnet.softplus(-x),
            -T.nnet.softplus(x))


def build_layer_outputs(P,name,layer_sizes,output_size):
    
    P["b_%s"%name] = np.zeros((output_size,),dtype=np.float32)
    b = P["b_%s"%name]
    def build_output(layer_no,input_size):
        P["W_output_%s_%d"%(name,layer_no)] = np.zeros((input_size,output_size),dtype=np.float32)
        W = P["W_output_%s_%d"%(name,layer_no)]
        return lambda x:T.nnet.softmax(T.dot(x,W) + b)

    output_transforms = [ build_output(i,size) for i,size in enumerate(layer_sizes) ]
    gate_transforms = [
            feedforward.build_transform(
                    P,"gate_%d"%i,
                    size,1,
                    initial_weights=lambda x,y:np.zeros((x,y)),
                    activation=lambda x: log_sigmoid(x)
                ) for i,size in enumerate(layer_sizes[:-1]) ]


    def gate_output_pairs(hiddens):
        outputs = [ ot(h) for ot,h in zip(output_transforms,hiddens) ]
        gates   = [ g(h)  for g, h in zip(gate_transforms,hiddens[:-1]) ]

        acc_log_neg_gates = 0
        acc_log_gates = [None] * len(layer_sizes)
        for i,(log_g,log_neg_g) in enumerate(gates):
            acc_log_gates[i]  = acc_log_neg_gates + log_g
            acc_log_neg_gates = acc_log_neg_gates + log_neg_g
        acc_log_gates[-1] = acc_log_neg_gates

        return acc_log_gates,outputs


    return gate_output_pairs


def build(P,input_size,layer_sizes,output_size):
    name = "classifier"
    activation = T.nnet.sigmoid
    initial_weights = feedforward.initial_weights
    output_hidden_size = 1024
    stopout_sizes = ( len(layer_sizes) - 1 ) * [output_hidden_size] + [layer_sizes[-1]]

    input_layer = feedforward.build_combine_transform(
            P,"%s_input"%name,
            [input_size],layer_sizes[0],
            initial_weights=initial_weights,
            activation=activation
        )

    transforms = feedforward.build_stacked_transforms(
            P,name,layer_sizes,
            initial_weights=initial_weights,
            activation=activation
        )

    gate_output_pairs = build_layer_outputs(
            P,name,
            stopout_sizes,
            output_size
        )

    def predict(X):
        hiddens = transforms(input_layer([X]))
        stopout_hiddens = [
                h[:,-output_hidden_size:] for h in hiddens[:-1] ] + [hiddens[-1]]
        return gate_output_pairs(stopout_hiddens)

    def model_cost(X,Y):
        log_gate_probs, outputs = predict(X)
        log_gate_prior = - T.mean(sum(
                log_p for i,log_p in enumerate(log_gate_probs)
            ))

        crossentropies = [ T.nnet.categorical_crossentropy(o,Y) for o in outputs ]
        log_layer_joint = [ log_p - ce
                                for log_p,ce in izip(log_gate_probs,crossentropies) ]
        max_joint = reduce(T.maximum,log_layer_joint)
        log_prob = T.log(sum(T.exp(llj - max_joint)
                            for llj in log_layer_joint)) + max_joint

        return -T.mean(log_prob), log_gate_prior, [ T.mean(ce) for ce in crossentropies ]

    def model_log_dist(X):
        log_gate_probs, outputs = predict(X)
        log_layer_joint_dist = [ log_p.dimshuffle(0,'x') + T.log(o)
                                    for log_p,o in izip(log_gate_probs,outputs) ]
        max_joint_dist = reduce(T.maximum,log_layer_joint_dist)
        log_dist = T.log(sum(T.exp(lljd - max_joint_dist)
                                for lljd in log_layer_joint_dist)) + max_joint_dist
        return log_dist

    return predict, model_cost,model_log_dist
