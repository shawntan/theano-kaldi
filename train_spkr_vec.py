import sys
import logging,json

import theano
import theano.tensor as T
import trainer
if __name__ == "__main__":
    import config
    config.parser.description = \
            "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence(
            "validation_frames_files","Validation set frames file.")
    config.structure("speaker_structure","Structure of speaker model.")
    config.structure("acoustic_structure","Structure of acoustic model.")
    config.structure("decoder_structure","Structure of decoder.")
    config.file("utt2spk_file","Utterance to speaker mapping.")
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

import spkr_vae
from itertools import tee
def make_split_stream(frames_files,utt2spk_file):
    streams = []
    for frames_file in frames_files:
        stream1, stream2 = tee(data_io.stream_file(frames_file))
        frame_stream = data_io.context(stream1,left=5,right=5)
        spkr_stream = data_io.augment_speaker_id(stream2,utt2spk_file)
        streams.append(data_io.zip_streams(frame_stream,spkr_stream))
    return streams


def build_data_stream(context=5):
    def data_stream(file_sequences):
        frames_files = file_sequences[0]
        split_streams = make_split_stream(frames_files,config.args.utt2spk_file)
        stream = data_io.random_select_stream(*split_streams)
        stream = data_io.buffered_random(stream)
        stream = data_io.randomise(stream)
        return stream
    return data_stream

def count_frames(labels_files,utt2spk_file):
    split_streams = [
            data_io.augment_speaker_id(data_io.stream(f,with_name=True),utt2spk_file) 
            for f in labels_files 
        ]
    frame_count = 0
    for _,l in chain(*split_streams):
        frame_count += l.shape[0]
    return frame_count

if __name__ == "__main__":

    input_size = config.args.speaker_structure[0]
    speaker_layer_sizes  = config.args.speaker_structure[1:-1]
    acoustic_layer_sizes = config.args.acoustic_structure[1:-1]
    decoder_layer_sizes = config.args.decoder_structure[:-1]
    speaker_latent_size = config.args.speaker_structure[-1]
    acoustic_latent_size = config.args.acoustic_structure[-1]

    training_frame_count = \
            count_frames(config.args.X_files,config.args.utt2spk_file)
    logging.debug("Created shared variables")


    P = Parameters()
    training_cost, speaker_vectors, acoustic_vectors = spkr_vae.build(
            P,
            x_size=input_size,
            acoustic_latent_size=acoustic_latent_size,
            speaker_latent_size=speaker_latent_size,
            speaker_layer_sizes=speaker_layer_sizes,
            acoustic_layer_sizes=acoustic_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            speaker_count=83
        )
    parameters = P.values()

    acoustic_latent_cost,speaker_latent_cost,\
                    speaker_prior_cost, recon_cost = training_cost(X,Y)
    training_loss = acoustic_latent_cost + speaker_latent_cost + recon_cost + speaker_prior_cost
#                    (0.5/training_frame_count) * sum(T.sum(T.sqr(w)) for w in parameters)
    monitored_values = {
            "training_loss": training_loss,
            "acoustic_latent_cost": acoustic_latent_cost,
            "speaker_latent_cost": speaker_latent_cost,
            "speaker_prior_cost": speaker_prior_cost,
            "recon_cost": recon_cost
        }
    monitored_keys = monitored_values.keys()

    test = theano.function(
            inputs = [X,Y],
            outputs = [ monitored_values[k] for k in monitored_keys ]
        )


    if config.args.pretrain_file != "":
        P.load(config.args.pretrain_file)

    logging.info("Parameters to tune:" + ','.join(w.name for w in parameters))


    loss = training_loss
    logging.debug("Built model expression.")

    logging.debug("Compiling functions...")
    update_vars = Parameters()
    gradients = T.grad(loss,wrt=parameters)

    logging.debug("Done.")

    run_train = compile_train_epoch(
            parameters,gradients,update_vars,
            data_stream=build_data_stream(context=5),
            update_strategy=updates.adam,
#            outputs=loss
#           outputs=[z_divergence,x_recon_cost,y_recon_cost,classification_loss]
#           outputs=[cross_entropy,cross_entropy_test]#[ T.sqrt(T.sum(w**2)) for w in gradients ]
        )

    def run_test():
        total_errors = None
        total_frames = 0
        split_streams = make_split_stream(
                config.args.validation_frames_files,
                config.args.utt2spk_file
            )
        for f,l in chain(*split_streams):
            if total_errors is None:
                total_errors = np.array(test(f,l),dtype=np.float32)
            else:
                total_errors += [f.shape[0] * v for v in test(f,l)]
            total_frames += f.shape[0]
        values = total_errors/total_frames
        return { k:float(v) for k,v in zip(monitored_keys,values) }

    train_loop(logging,run_test,run_train,P,update_vars,monitor_score="training_loss")
