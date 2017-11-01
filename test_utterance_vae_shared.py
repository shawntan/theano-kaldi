import theano.tensor as T
import theano
import sys
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

import data_io
import config
import logging
import utterance_frame_data
import utterance_vae_conv

import epoch_train_loop
import validator

PRINT_GRADIENTS = False


@config.option("gradient_clip", "Gradient magnitude clipping.",
               type=config.float)
@config.option("learning_rate", "Learning rate.", type=config.float)
def build_updates(parameters, gradients, update_vars,
                  gradient_clip, learning_rate):
#    gradients = updates.clip_deltas(gradients, gradient_clip)
    return updates.adam(parameters, gradients,
                        learning_rate=learning_rate,
                        P=update_vars)


@config.option("iteration_log", "File for dumping iteration cost.",
               default=None)
def build_train_function(inputs, outputs, updates, iteration_log):
    logging.info("Trying to compile...")
    train_fn = theano.function(
        inputs=inputs,
        outputs=(outputs if iteration_log is not None else None),
        updates=updates,
    )

    if iteration_log is not None:
        if iteration_log == '-':
            fd = sys.stdout
        else:
            fd = open(iteration_log, 'w', 0)

        def train(args):
            values = train_fn(*args)
            if PRINT_GRADIENTS:
                from pprint import pprint
                pprint(values)
                if any(np.isnan(values[k]).any() for k in values):
                    exit()
            else:
                print >> fd, ' '.join(str(v) for v in values)
    else:
        def train(args):
            train_fn(*args)

    return train


@config.option("temporary_model_file",
               "Temporary file to store model parameters during training.")
@config.option("temporary_training_file",
               "Temporary file to store training parameters during training.")
def build_validation_callback(P, update_vars, temporary_model_file,
                              temporary_training_file):
    def validation_callback(prev_score, curr_score):
        if curr_score < prev_score:
            logging.info("Better than before, saving model.")
            P.save(temporary_model_file)
            update_vars.save(temporary_training_file)
    return validation_callback


@config.option("previous_model_file", "Previous train state.", default="")
@config.option("previous_training_file", "Previous train state.", default="")
def load_previous_train_state(P, update_vars, previous_model_file,
                              previous_training_file):
    if previous_model_file != "":
        P.load(previous_model_file)
        update_vars.load(previous_training_file)
        logging.debug("Previous train state loaded.")


if __name__ == "__main__":
    config.parse_args()
    P = Parameters()
    training_costs = utterance_vae_conv.build(P)

    X = T.tensor3('X')
    utt_lengths = T.ivector('utt_lengths')

    speaker_latent_cost, acoustic_latent_cost, recon_cost = \
        training_costs(X, utt_lengths)

    parameters = P.values()
    logging.info("Tuning parameters: " + ", ".join(w.name for w in parameters))

    _utt_lengths = T.cast(utt_lengths, 'float32')

    f = theano.function(
        inputs=[X, utt_lengths],
        outputs={
            "speaker_latent": speaker_latent_cost,
            "acoustic_latent": acoustic_latent_cost,
            "recon": recon_cost
        }
    )

    # P.load('exp/vae_fbank/calibrating_vae/fbank.mila/generative.pkl.tmp')

    from pprint import pprint
    for x, lengths in utterance_frame_data.batched_training_stream():
        batched_output = f(x, lengths)
        single_output = f(x[:1, :lengths[0]], lengths[:1])
        print
        print single_output['speaker_latent'][0], single_output['acoustic_latent'][0], single_output['recon'][0]
        print batched_output['speaker_latent'][0], batched_output['acoustic_latent'][0], batched_output['recon'][0]

        batched_output = f(x, lengths)
        single_output = f(x[:1, :lengths[0]], lengths[:1])
        print single_output['speaker_latent'][0], single_output['acoustic_latent'][0], single_output['recon'][0]
        print batched_output['speaker_latent'][0], batched_output['acoustic_latent'][0], batched_output['recon'][0]

        break
