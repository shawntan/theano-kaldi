import sys
import logging,json

import theano
import theano.tensor as T
import trainer
if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.file_sequence("validation_labels_files","Validation set labels file.")
    config.structure("speaker_structure","Structure of speaker model.")
    config.structure("acoustic_structure","Structure of acoustic model.")
    config.structure("decoder_structure","Structure of decoder.")
    config.file("pretrain_file","Pretrain file.",default="")

    X = T.matrix('X')
    Y = T.ivector('Y')
    compile_train_epoch = trainer.build_train_epoch([X,Y])
    train_loop = trainer.build_train_loop()

    config.parse_args()
import numpy as np
import math

import data_io
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import sep_vae

def make_split_stream(frames_files,labels_files):
    return [ data_io.zip_streams(
                data_io.context(
                    data_io.stream_file(frames_file),
                    left=5,right=5
                ),
                data_io.stream_file(labels_file)
            ) for frames_file,labels_file in izip(frames_files,labels_files) ]


def build_data_stream(context=5):
    def data_stream(file_sequences):
        frames_files = file_sequences[0]
        labels_files = file_sequences[1]
        split_streams = make_split_stream(frames_files,labels_files) 
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        stream = data_io.randomise(stream)
        return stream
    return data_stream

def count_frames(labels_files,output_size):
    split_streams = [ data_io.stream(f) for f in labels_files ]
    frame_count = 0
    label_count = np.zeros(output_size,dtype=np.float32)
    for l in chain(*split_streams):
        frame_count += l.shape[0]
        np.add.at(label_count,l,1)
    return frame_count,label_count

if __name__ == "__main__":
    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]
 
    training_frame_count, training_label_count = count_frames(config.args.Y_files,output_size)
    logging.debug("Created shared variables")


    P = Parameters()
    training_cost,classification_cost = sep_vae.build(P,
            input_size,[2048,2048,2048,2048],64,
            output_size,[2048]
        )
    logging.debug(str(training_label_count))
    P['b_y_model_decoder_output'].set_value(np.log(training_label_count))

    prior_cost,z_divergence,x_recon_cost,y_recon_cost = training_cost(X,Y)
    training_loss = prior_cost + z_divergence + x_recon_cost + y_recon_cost
    classification_loss, zero_one_loss  = classification_cost(X,Y)

    monitored_values = {
            "training_loss": training_loss,
            "z_divergence":z_divergence,
            "x_recon_cost":x_recon_cost,
            "y_recon_cost":y_recon_cost,
            "cross_entropy":classification_loss,
            "errors":zero_one_loss
        }
    monitored_keys = monitored_values.keys()

    test = theano.function(
            inputs = [X,Y],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )


    if config.args.pretrain_file != "":
        P.load(config.args.pretrain_file)

    parameters = P.values() 
    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))


    loss = training_loss + \
            (0.5/training_frame_count) * sum(T.sum(T.sqr(w)) for w in parameters)
    logging.debug("Built model expression.")

    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)

    logging.debug("Done.")

#    def strat(parameters, gradients, learning_rate=1e-3, norm_ceil=10, P=None):
#        norm = T.sqrt(sum(T.sum(T.sqr(g)) for g in gradients))
#        factor = T.switch(norm < norm_ceil,1,norm_ceil/norm)
#        gradients = [ g * factor for g in gradients ]
#        return updates.adam(parameters,gradients,learning_rate=learning_rate)

    run_train = compile_train_epoch(
            parameters,gradients,update_vars,
            data_stream=build_data_stream(context=5),
            update_strategy=updates.adam,
#           outputs=[z_divergence,x_recon_cost,y_recon_cost,classification_loss]
#           outputs=[cross_entropy,cross_entropy_test]#[ T.sqrt(T.sum(w**2)) for w in gradients ]
        )

    def run_test():
        total_errors = None
        total_frames = 0
        split_streams = make_split_stream(config.args.validation_frames_files,
                                            config.args.validation_labels_files) 
        for f,l in chain(*split_streams):
            if total_errors is None:
                total_errors = np.array(test(f,l),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f,l)]
            total_frames += f.shape[0]
        values = total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    train_loop(logging,run_test,run_train,P,update_vars,monitor_score="cross_entropy")
