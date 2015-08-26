import sys
import logging,json
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.file_sequence("validation_frames_files","Validation set frames file.")
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
import data_io
import model
import cPickle as pickle
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
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

    frames_files     = config.args.frames_files
    val_frames_files = config.args.validation_frames_files
    minibatch_size = config.args.minibatch

    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]

    P = Parameters()
    feedforward = model.build_feedforward(
            P,
            input_size = input_size,
            layer_sizes = layer_sizes,
            output_size = output_size
        )

    X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
    X = T.matrix('X')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')

    layers,_ = feedforward(X)


    layer_sizes = [input_size] + layer_sizes[:-1]
    pretrain_functions = []
    for i,layer in enumerate(layers[:-1]):
        logging.debug("Compiling functions for layer %d"%i)
        W = P["W_hidden_%d"%i]
        b = P["b_hidden_%d"%i]
        b_rec = theano.shared(
                np.zeros((layer_sizes[i],),dtype=theano.config.floatX),
                name="b_rec_%d"
            )
        loss = cost(
                layer,
                reconstruct(
                    corrupt(layer),
                    W,b,b_rec,
                    input_layer = (layer.name == 'X')
                ),
                kl_divergence = (layer.name != 'X')
            )
        lr = 0.01 if i > 0 else 0.003
        parameters = [W,b,b_rec]
        gradients  = T.grad(loss,wrt=parameters)
        train = theano.function(
                inputs = [start_idx,end_idx],
                outputs = loss,
                updates = updates.momentum(parameters,gradients,learning_rate=lr),
                givens  = {
                    X: X_shared[start_idx:end_idx],
                }
            )
        test = theano.function(inputs=[X],outputs=loss)

        pretrain_functions.append((train,test))
        logging.debug("Done compiling for layer %d"%i)

    def run_train(train):
        split_streams = [ data_io.stream(f) for f in frames_files ]
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        total_frames = 0
        for f,size in data_io.randomise(stream):
            X_shared.set_value(f)
            batch_count = int(math.ceil(size/float(minibatch_size)))
            for idx in xrange(batch_count):
                start = idx*minibatch_size
                end = min((idx+1)*minibatch_size,size)
                train(start,end)

    def run_test(test):
        total_errors = 0
        total_frames = 0
        split_streams = [ data_io.stream(f) for f in val_frames_files ]
        for f in chain(*split_streams):
            total_errors += f.shape[0] * test(f)
            total_frames += f.shape[0]
        values = total_errors / total_frames
        return values



    for layer_idx,(train,test) in enumerate(pretrain_functions):    
        logging.debug("Pretraining layer " + str(layer_idx) + "...")
        best_score = np.inf
        for epoch in xrange(config.args.max_epochs):
            run_train(train)
            score = run_test(test)
            logging.debug("Score on validation set: " + str(score))
            if score < best_score:
                best_score = score
                logging.debug("Saving model.")
                P.save(config.args.temporary_file) 
            else:
                logging.debug("Layer done.")
                P.load(config.args.temporary_file) 
                break
    model.save(config.args.output_file,params)
