if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.structure("structure","Structure of discriminative model.")
    config.file("output_file","Output file.")
    config.file("temporary_file","Temporary file.")
    config.integer("minibatch","Minibatch size.",default=128)
    config.integer("max_epochs","Maximum number of epochs to train.",default=20)
    config.parse_args()
import theano
import theano.tensor as T

import numpy as np
import math
import sys

import data_io
import model
import updates
import cPickle as pickle
from itertools import izip, chain

theano_rng = T.shared_randomstreams.RandomStreams(np.random.RandomState(1234).randint(2**30))

def corrupt(x,corr=0.2):
    corr_x = theano_rng.binomial(size=x.shape,n=1,p=1-corr,dtype=theano.config.floatX) * x
    corr_x.name = "corr_" + x.name
    return corr_x

def reconstruct(corr_x,W,b,b_rec,input_layer):
    hidden = T.nnet.sigmoid(T.dot(corr_x,W) + b)

    recon  = T.dot(hidden,W.T) + b_rec
    if not input_layer:
        recon = T.nnet.sigmoid(recon)
    return recon

def cost(x,recon_x,kl_divergence):
    if not kl_divergence:
        return T.mean(T.sum((x - recon_x)**2,axis=1))
    else:
        return -T.mean(T.sum(x * T.log(recon_x) + (1 - x) * T.log(1 - recon_x), axis=1))

if __name__ == "__main__":

    frames_files   = config.args.frames_files
    labels_files   = config.args.labels_files
    minibatch_size = config.args.minibatch

    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]

    params = {}

    feedforward = model.build_feedforward(params)

    X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
    X = T.matrix('X')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')

    layers,_ = feedforward(X)


    layer_sizes = [input_size] + layer_sizes[:-1]
    pretrain_functions = []
    for i,layer in enumerate(layers[:-1]):
        W = params["W_hidden_%d"%i]
        b = params["b_hidden_%d"%i]
        b_rec = theano.shared(np.zeros((layer_sizes[i],),dtype=theano.config.floatX))

        loss = cost(
                layer,
                reconstruct(
                    corrupt(layer),
                    W,b,b_rec,
                    input_layer = (layer.name == 'X')
                ),
                kl_divergence = ((layer.name != 'X')
            )
        lr = 0.01 if i > 0 else 0.003
        parameters = [W,b,b_rec]
        gradients  = T.grad(loss,wrt=parameters)
        train = theano.function(
                inputs = [start_idx,end_idx],
                outputs = loss,
                updates = updates.momentum(parameters,gradients,eps=lr),
                givens  = {
                    X: X_shared[start_idx:end_idx],
                }
            )
        pretrain_functions.append(train)

    for layer_idx,train in enumerate(pretrain_functions):    
        print "Pretraining layer",layer_idx,"..."
        for epoch in xrange(config.args.max_epochs):    
            split_streams = [ data_io.stream(f,l) for f,l in izip(frames_files,labels_files) ]
            stream = chain(*split_streams)
            total_count = 0
            total_loss  = 0
            for f,_,size in data_io.randomise(stream):
                X_shared.set_value(f)
                batch_count = int(math.ceil(size/float(minibatch_size)))
                for idx in xrange(batch_count):
                    start = idx*minibatch_size
                    end = min((idx+1)*minibatch_size,size)
                    total_loss += train(start,end)
                    total_count += 1
            print total_loss/total_count
    model.save(config.args.output_file,params)
