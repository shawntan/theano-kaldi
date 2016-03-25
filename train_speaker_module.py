import sys
import logging,json

import theano
import theano.tensor as T
import trainer

if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.structure("speaker_structure","Structure of speaker model.")
    config.structure("structure","Structure of discriminative model.")
    config.file("canonical_model","Pickle file containing canonical model.")
    config.file("vae_model","Pickle file containing vae model.")
    config.file("output","Pickle file containing final model.")
    config.file("pooling_method","Method for pooling across utterance")
    config.file_sequence("frames_files","Frames files.")
    config.file_sequence("labels_files","Labels files.")
    config.file_sequence("validation_frames_files","Validation frames files.")
    config.file_sequence("validation_labels_files","Validation labels files.")
    config.parse_args()

import numpy as np
import math

import data_io
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

from itertools import tee

from train import *
import model
import utterance_vae
def make_split_stream(frames_files,labels_files,speaker_representation):
    streams = []
    for frames_file,labels_files in zip(frames_files,labels_files):
        frame_stream = data_io.context(data_io.stream_file(frames_file),left=5,right=5)
        frame_stream1, frame_stream2 = tee(frame_stream)
        spkr_stream = ((n,np.tile(speaker_representation(f),(f.shape[0],1))) 
                            for n,f in frame_stream2)
        label_stream = data_io.stream_file(labels_files)
        stream = data_io.zip_streams(frame_stream1,spkr_stream,label_stream)
        streams.append(stream)
    return streams

def data_stream(frames_files,labels_files,speaker_representation):
    split_streams = make_split_stream(frames_files,labels_files,speaker_representation)
    split_streams = [ data_io.chunk(data_io.buffered_random(s)) 
                        for s in split_streams ]
    stream = data_io.random_select_stream(*split_streams)
    stream = data_io.buffered_random(stream)
    stream = data_io.randomise(stream)
    return stream

if __name__ == "__main__":
    input_size  = config.args.structure[0]
    layer_sizes = config.args.structure[1:-1]
    output_size = config.args.structure[-1]
    speaker_layer_sizes  = config.args.speaker_structure[1:-1]
    speaker_latent_size = config.args.speaker_structure[-1]

    X = T.matrix('X')
    Y = T.ivector('Y')
    speaker = T.matrix('speaker')
    P_speakers = Parameters()
    speaker_encode = utterance_vae.build_speaker_inferer(
            P_speakers,method=config.args.pooling_method,
            x_size=input_size,
            speaker_latent_size=speaker_latent_size,
            speaker_layer_sizes=speaker_layer_sizes
        )
    speaker_vector,_,_ = speaker_encode(X.dimshuffle('x',0,1))
    speaker_representation = theano.function(
            inputs=[X],
            outputs=speaker_vector[0]
        )


    P = Parameters()
    classify = model.build_with_speaker_module(
            P,32,input_size,layer_sizes,output_size)


    _, outputs = classify(X,speaker)
    cross_entropy = T.mean(crossentropy(outputs,Y))
    outputs_test = outputs
    cross_entropy_test = cross_entropy

    parameters = P.values()
#    parameters = [ P.W_classifier_input_1, P.b_classifier_input ]
    minibatch_size = 512
    loss = cross_entropy

    monitored_values = {
            "cross_entropy": cross_entropy_test,
            "classification_error":T.mean(T.neq(T.argmax(outputs_test,axis=1),Y))
        }

    monitored_keys = monitored_values.keys()
    P.load(config.args.canonical_model)
    P_speakers.load(config.args.vae_model)
    logging.info("Parameters to tune: " + ','.join(w.name for w in parameters))


    logging.debug("Built model expression.")

    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)
    logging.debug("Compiling functions...")

    test = theano.function(
            inputs = [X,Y,speaker],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )

    import trainer
    X_shared = trainer.define_shared_variable(X)
    speaker_shared = trainer.define_shared_variable(speaker)
    Y_shared = trainer.define_shared_variable(Y)
    lr = T.scalar('lr')
    start_idx = T.iscalar('start_idx')
    end_idx = T.iscalar('end_idx')
    train = theano.function(
            inputs = [lr,start_idx,end_idx],
            #outputs = [cross_entropy_test, T.sum(T.sqr(P.W_classifier_input_1))],
            #updates = updates.adam(parameters,gradients,P=update_vars,learning_rate=lr),
            #updates = updates.momentum(parameters,gradients,P=update_vars,learning_rate=lr),
            updates = [ (p,p - lr * g) for p,g in zip(parameters,gradients) ],
            givens = {
                X: X_shared[start_idx:end_idx],
                Y: Y_shared[start_idx:end_idx],
                speaker: speaker_shared[start_idx:end_idx]
            }
        )
    def run_test():
        total_errors = None
        total_frames = 0
        split_streams = make_split_stream(
                config.args.validation_frames_files,
                config.args.validation_labels_files,
                speaker_representation
            )

        for f,s,l in chain(*split_streams):
            if total_errors is None:
                total_errors = np.array(test(f,l,s),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f,l,s)]
            total_frames += f.shape[0]
        values = total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    logging.debug("Done.")
    minibatch_size = 512
    best_score = np.inf
    for _ in xrange(50):
        stream = data_stream(config.args.frames_files,
                             config.args.labels_files,speaker_representation)
        logging.debug("Running test.")

        results = run_test()
        logging.info(str(results))
        if best_score > results['cross_entropy']:
            logging.debug('Best score, saving model.')
            best_score = results['cross_entropy']
            P.save(config.args.output)
         
        for item in stream:
            size = item[-1]
            batch_count = int(math.ceil(size/float(minibatch_size)))
            X_shared.set_value(item[0])
            speaker_shared.set_value(item[1])
            Y_shared.set_value(item[2])
            for idx in xrange(batch_count):
                start = idx*minibatch_size
                end = min((idx+1)*minibatch_size,size)
                train(0.05,start,end)

    results = run_test()
    if best_score > results['cross_entropy']:
        logging.debug('Best score, saving model.')
        best_score = results['cross_entropy']
        P.save(config.args.output)
    print results
