import sys
import logging,json
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.file_sequence("validation_frames_files","Validation set frames file.")

    config.structure("structure","Structure of discriminative model.")
    config.file("output_file","Output file.")
    config.file("temporary_file","Temporary file.")
    config.integer("minibatch","Minibatch size.",default=512)
    config.integer("max_epochs","Maximum number of epochs to train.",default=200)
    config.parse_args()

import theano
import theano.tensor as T

import numpy as np
import math

import data_io
import model
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import vae
if __name__ == "__main__":
    frames_files    = config.args.frames_files
    val_frames_files = config.args.validation_frames_files
    minibatch_size  = config.args.minibatch
    input_size      = config.args.structure[0]
    layer_sizes     = config.args.structure[1:-1]
    output_size     = config.args.structure[-1]
    
    logging.info("Training data:     " + ','.join(frames_files))
    logging.info("Validation data:   " + ','.join(val_frames_files))
    logging.info("Minibatch size:    " + str(minibatch_size))
    logging.info("Structure:         " + ':'.join(map(str,config.args.structure)))
    
    X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
    logging.debug("Created shared variables")

    P = Parameters()

    _,_,recon_error = model.build_unsupervised(P,input_size,layer_sizes,output_size)

    X = T.matrix('X')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    lr = T.scalar('lr')

    recon_X,cost,kl_divergence,log_p = recon_error(X)

    parameters = P.values() 
    loss = cost #+ 0.5 * sum(T.sum(T.sqr(w)) for w in parameters)
    logging.debug("Built model expression.")

    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))
    
    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)

    train = theano.function(
            inputs  = [lr,start_idx,end_idx],
            outputs = cost,
            updates = updates.momentum(parameters,gradients,learning_rate=lr,P=update_vars),
            givens  = {
                X: X_shared[start_idx:end_idx],
            }
        )
    
    monitored_values = {
            "loss": cost,
            "squared_error":T.mean(T.sum(T.sqr(recon_X - X),axis=1),axis=0),
            "neg_log_p": -log_p,
            "kl_divergence": kl_divergence
        }
    monitored_keys = monitored_values.keys()
    test = theano.function(
            inputs = [X],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )

    logging.debug("Done.")

    def run_test():
        total_errors = None
        total_frames = 0
        split_streams = [ data_io.stream(f) for f in val_frames_files ]
        for f in chain(*split_streams):
            if total_errors is None:
                total_errors = np.array(test(f),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f)]
            total_frames += f.shape[0]
        values = total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    def run_train():
        split_streams = [ data_io.stream(f) for f in frames_files ]
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)

        total_frames = 0
        for f,size in data_io.randomise(stream):
            total_frames += f.shape[0]
            X_shared.set_value(f)
            batch_count = int(math.ceil(size/float(minibatch_size)))
            for idx in xrange(batch_count):
                start = idx*minibatch_size
                end = min((idx+1)*minibatch_size,size)
                train(learning_rate,start,end)

    
    learning_rate = 5e-5
    best_score = np.inf
    
    logging.debug("Starting training process...")
    for epoch in xrange(config.args.max_epochs):
        scores = run_test()
        score = scores['loss']
        logging.info("Epoch " + str(epoch) + " results: " + json.dumps(scores))
        _best_score = best_score

        if score < _best_score:
            logging.debug("score < best_score, saving model.")
            best_score = score
            P.save(config.args.temporary_file)
            update_vars.save("update_vars.tmp")

        if score/_best_score > 0.999 and epoch > 0:
            learning_rate *= 0.5
            logging.debug("Halving learning rate. learning_rate = " + str(learning_rate))
            logging.debug("Loading previous model.")
            P.load(config.args.temporary_file)
            update_vars.load("update_vars.tmp")

        if learning_rate < 1e-9: break
        
        logging.info("Epoch %d training."%(epoch + 1))
        run_train()
        logging.info("Epoch %d training done."%(epoch + 1))

    P.load(config.args.temporary_file)
    P.save(config.args.output_file)
    logging.debug("Done training process.")
