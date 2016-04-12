import sys
import logging,json

import theano
import theano.tensor as T
import trainer
import config
if __name__ == "__main__":
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.file_sequence("validation_labels_files","Validation set labels file.")
    config.structure("structure","Structure of discriminative model.")
    config.file("pretrain_file","Pretrain file.",default="")
    config.file("vae_model","Pretrain file.",default="")
    config.real("momentum","Momentum.",default=0.9)

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

import feedforward


encode = None
def feature_generator():
    global encode
    if encode is None:
        import utterance_vae
        P = Parameters()
        speaker_encode, acoustic_encode = utterance_vae.build_encoder(
                P,pooling_method="max",
                x_size=440,
                acoustic_latent_size=64,
                speaker_latent_size=32,
                speaker_layer_sizes=[1024,1024],
                acoustic_layer_sizes=[1024,1024]
            )
        X = T.matrix('X')
        X_ = X.dimshuffle('x',0,1)
        utterance_speaker,_,_ = speaker_encode(X_)
        utterance_speaker = utterance_speaker.dimshuffle(0,'x',1)
        acoustic, _, _= acoustic_encode([X_,utterance_speaker])
        #logging.debug("Compiling encoder...")
        encode = theano.function(
                inputs=[X],
                outputs=T.concatenate([
                    acoustic[0],
                    T.tile(utterance_speaker[0,0],(acoustic[0].shape[0],1))
                ],axis=1)
            )
        P.load(config.args.vae_model)
        #print encode(np.random.randn(100,440).astype(np.float32))
    return encode

def make_split_stream(frames_files,labels_files):
    streams = []
    encoder = feature_generator()
    for frames_file,labels_file in zip(frames_files,labels_files):
        frame_stream = data_io.stream_file(frames_file)
        frame_stream = data_io.context(frame_stream,left=5,right=5)
        frame_stream = ((n,encoder(f)) for n,f in frame_stream)
        label_stream = data_io.stream_file(labels_file)
        stream = data_io.zip_streams(frame_stream,label_stream)
        streams.append(stream)
    return streams

def build_data_stream(context=5):
    def data_stream(file_sequences):
        frames_files = file_sequences[0]
        labels_files = file_sequences[1]
        split_streams = make_split_stream(frames_files,labels_files) 
        split_streams = [ data_io.chunk(data_io.buffered_random(s)) 
                            for s in split_streams ]
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        stream = data_io.async(stream)
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

def crossentropy(output,Y):
    if output.owner.op == T.nnet.softmax_op:
        x = output.owner.inputs[0]
        k = T.max(x,axis=1,keepdims=True)
        sum_x = T.log(T.sum(T.exp(x - k),axis=1)) + k
        return - x[T.arange(x.shape[0]),Y] + sum_x
    else:
        return T.nnet.categorical_crossentropy(outputs,Y)

if __name__ == "__main__":
    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]

    training_frame_count, training_label_count = count_frames(config.args.Y_files,output_size)
    logging.debug("Created shared variables")


    P = Parameters()
    classify = feedforward.build_classifier(
            P,"classifier",
            [input_size],layer_sizes,output_size,
            initial_weights=feedforward.initial_weights,
            activation=T.nnet.sigmoid
        )

    _, outputs = classify([X])
    cross_entropy = T.mean(crossentropy(outputs,Y))

    monitored_values = {
            "cross_entropy": cross_entropy,
            "classification_error": T.mean(T.neq(T.argmax(outputs,axis=1),Y))
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


    loss = (cross_entropy + \
            (0.5/training_frame_count) * sum(T.sum(T.sqr(w)) for w in parameters))
    logging.debug("Built model expression.")

    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)

    logging.debug("Done.")

    def strat(parameters, gradients, learning_rate=1e-3, P=None):
#        return [ (p, p - learning_rate * g) 
#                    for p,g in zip(parameters,gradients) ]
        return updates.momentum(
                parameters,gradients,
                mu=config.args.momentum,
                learning_rate=learning_rate/T.cast(X.shape[0],'float32'),
                P=P
            )

    run_train = compile_train_epoch(
            parameters,gradients,update_vars,
            data_stream=build_data_stream(context=5),
            update_strategy=strat,
            outputs=cross_entropy #[ T.sqrt(T.sum(w**2)) for w in gradients ]
        )

    def run_test():
        values = 0
        for i in xrange(1):
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
            values = values + total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    P.W_classifier_output.set_value(0 * P.W_classifier_output.get_value())
    P.b_classifier_output.set_value(np.log(training_label_count))
    train_loop(logging,run_test,run_train,P,update_vars,monitor_score="cross_entropy")
