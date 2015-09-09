import sys
import logging,json
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("frames_files",".pklgz file containing audio frames.")
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.structure("structure_z1","Structure of M1.")
    config.structure("structure_y","Structure of labeled part of M2.")
    config.structure("structure_z2","Structure of latent var part of M2.")
    config.structure("structure_z1_recon","Structure of decode part of M2.")
    config.file("z1_file","Z1 params file.")
    config.file("output_file","Output file.")
    config.file("temporary_file","Temporary file.")
    config.file("utt2spk_file","utt2spk file.")
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

    z1_input_size      = config.args.structure_z1[0]
    z1_layer_sizes     = config.args.structure_z1[1:-1]
    z1_output_size     = config.args.structure_z1[-1]
    
    y_layer_sizes     = config.args.structure_y[:-1]
    y_output_size     = config.args.structure_y[-1]

    z2_layer_sizes     = config.args.structure_z2[:-1]
    z2_output_size     = config.args.structure_z2[-1]

    z1_recon_sizes = config.args.structure_z1_recon


    logging.info("Training data:     " + ', '.join(frames_files))
    logging.info("Validation data:   " + ', '.join(val_frames_files))
    logging.info("Minibatch size:    " + str(minibatch_size))
    logging.info(z1_recon_sizes)
    
    X_shared = theano.shared(np.zeros((1,z1_input_size),dtype=theano.config.floatX))
    S_shared = theano.shared(np.zeros((1,),dtype=np.int32))
    logging.debug("Created shared variables")

    P_z1_x = Parameters()
    encode_Z1,_,_ = model.build_unsupervised(P_z1_x,z1_input_size,z1_layer_sizes,z1_output_size)
    P_z1_x.load(config.args.z1_file)

    P = Parameters()
    infer_Y,_,Z1_recon_error,Z1_recon_error_unseen = model.build_Y_Z2_generative(P,
                input_size=z1_output_size,
                y_hidden_sizes=y_layer_sizes,
                y_output_size=y_output_size,
                z2_hidden_sizes=z2_layer_sizes,
                z2_output_size=z2_output_size,
                z1_decoder_sizes=z1_recon_sizes
            )
    X = T.matrix('X')
    S = T.ivector('S')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    lr = T.scalar('lr')
    
    Z1,_,_ = encode_Z1([X])
    recon_Z1,classification_error,cost = Z1_recon_error(S,Z1)
    _,cost_unseen = Z1_recon_error_unseen(Z1)

    parameters = P.values() 
    loss = cost + 0.5 * sum(T.sum(T.sqr(w)) for w in parameters)
    logging.debug("Built model expression.")
    logging.info("Parameters to tune: " + ', '.join(w.name for w in parameters))
    
    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)

    train = theano.function(
            inputs  = [lr,start_idx,end_idx],
            outputs = [cost,T.mean(T.sum(T.sqr(Z1-recon_Z1),axis=1)),classification_error],
            updates = updates.momentum(parameters,gradients,learning_rate=lr,P=update_vars),
            givens  = {
                X: X_shared[start_idx:end_idx],
                S: S_shared[start_idx:end_idx],
            }
        )
    
    monitored_values = {
            "loss": cost_unseen,
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

    import sa_io
    def run_train():
        utt2spk = sa_io.utterance_speaker_mapping(config.args.utt2spk_file)
        split_streams = [ data_io.stream(f,with_name=True) for f in frames_files ]
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        stream = sa_io.frame_speaker_stream(stream,utt2spk)

        total_frames = 0
        for f,s,size in data_io.randomise(stream):
            total_frames += f.shape[0]
            X_shared.set_value(f)
            S_shared.set_value(s)
            batch_count = int(math.ceil(size/float(minibatch_size)))
            for idx in xrange(batch_count):
                start = idx*minibatch_size
                end = min((idx+1)*minibatch_size,size)
                print train(learning_rate,start,end)
    
    learning_rate = 1e-4
    best_score = np.inf
    
    logging.debug("Starting training process...")
    for epoch in xrange(config.args.max_epochs):
#        scores = run_test()
#        score = scores['loss']
#        logging.info("Epoch " + str(epoch) + " results: " + json.dumps(scores))
#        _best_score = best_score
#
#        if score < _best_score:
#            logging.debug("score < best_score, saving model.")
#            best_score = score
#            P.save(config.args.temporary_file)
#            update_vars.save("update_vars.tmp")
#
#        if score/_best_score > 0.999 and epoch > 0:
#            learning_rate *= 0.5
#            logging.debug("Halving learning rate. learning_rate = " + str(learning_rate))
#            logging.debug("Loading previous model.")
#            P.load(config.args.temporary_file)
#            update_vars.load("update_vars.tmp")
#
#        if learning_rate < 1e-8: break
        
        logging.info("Epoch %d training."%(epoch + 1))
        run_train()
        logging.info("Epoch %d training done."%(epoch + 1))

    P.load(config.args.temporary_file)
    P.save(config.args.output_file)
    logging.debug("Done training process.")
