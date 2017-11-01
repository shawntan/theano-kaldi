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

    per_utterance_latent_cost_est = T.mean(speaker_latent_cost, axis=0)
    per_utterance_acoustic_cost_est = T.mean(acoustic_latent_cost, axis=0)
    per_utterance_recon_cost_est = T.mean(recon_cost, axis=0)
    logging.info("Counting frames...")
    utterance_count = sum(1 for _ in utterance_frame_data.training_stream())
    frame_count = sum(x.shape[0]
                      for (x,) in utterance_frame_data.training_stream())
    logging.info("Total utterances: %d" % utterance_count)
    logging.info("Total frames: %d" % frame_count)
    avg_frames_per_utterance = frame_count / float(utterance_count)
    logging.info("Avg. frames / utterance: %0.2f" % avg_frames_per_utterance)

    norm_factor = T.cast(X.shape[0], 'float32') / T.sum(_utt_lengths)

    batch_frame_count = T.sum(_utt_lengths)
    per_frame_acoustic_cost_est = T.sum(
        acoustic_latent_cost) / batch_frame_count
    per_frame_recon_cost_est = T.sum(recon_cost) / batch_frame_count

    cost = ((T.mean(speaker_latent_cost, axis=0) / avg_frames_per_utterance) +
            per_frame_acoustic_cost_est +
            per_frame_recon_cost_est)

    cost_val = per_utterance_latent_cost_est + \
        per_utterance_acoustic_cost_est + \
        per_utterance_recon_cost_est

    l2_weight = 0.5 / float(frame_count)
    loss = (cost +
            l2_weight * sum(
                 T.sum(T.sqr(w)) for w in parameters
                 if w.name not in []
            ))

#    factor = np.sqrt(avg_frames_per_utterance)
#    P.W_acoustic_encoder_input_1.set_value(
#        P.W_acoustic_encoder_input_1.get_value() / factor)
#    P.W_decode_input_1.set_value(
#        P.W_decode_input_1.get_value() / factor)
#    P.W_speaker_encoder_pooled_mean.set_value(
#        P.W_speaker_encoder_pooled_mean.get_value() / factor)
#    P.W_speaker_encoder_pooled_std.set_value(
#        P.W_speaker_encoder_pooled_std.get_value() / factor)
#    P.W_acoustic_encoder_mean.set_value(
#            P.W_acoustic_encoder_mean.get_value() / 10)
#    P.W_acoustic_encoder_std.set_value(
#            P.W_acoustic_encoder_std.get_value() / 10)

    gradients = T.grad(loss, wrt=parameters)
    update_vars = Parameters()

    k = T.max([T.max(abs(w)) for w in gradients])
    grad_mag = T.sqrt(sum(T.sum(T.sqr(w / k)) for w in gradients)) * k

    deltas = gradients
    delta_sum = sum(T.sum(d) for d in deltas)
    not_finite = T.isnan(delta_sum) | T.isinf(delta_sum)
    if PRINT_GRADIENTS:
        outputs = {p.name: T.mean(abs(g))
                   for p, g in zip(parameters, gradients)}
        outputs['recon_cost'] = ((per_utterance_recon_cost_est * X.shape[0]) /
                                 T.sum(_utt_lengths))
    else:
        outputs = [
            per_utterance_latent_cost_est,
            (per_utterance_acoustic_cost_est *
             X.shape[0]) / T.sum(_utt_lengths),
            (per_utterance_recon_cost_est * X.shape[0]) / T.sum(_utt_lengths),
            T.mean(_utt_lengths),
            grad_mag,
            not_finite,
            loss,
        ]
    logging.info("Compiling training function")
    train = build_train_function(
        inputs=[X, utt_lengths],
        outputs=outputs,
        updates=build_updates(parameters, gradients, update_vars=update_vars)
    )

    logging.info("Compiling validation function")
    validate = validator.build(
        inputs=[X, utt_lengths],
        outputs={
            "utterance_latent": per_utterance_latent_cost_est,
            "frame_latent": per_utterance_acoustic_cost_est,
            "recon_cost": per_utterance_recon_cost_est,
            "total": cost_val
        },
        monitored_var="total",
        validation_stream=utterance_frame_data.validation_stream,
        callback=build_validation_callback(P, update_vars)
    )

    def validate_report(x):
        logging.info("Epoch %d done." % x)
        report = validate()
        logging.info(report)
    logging.info("Trying to load previous train state..")
    load_previous_train_state(P, update_vars)

    logging.info("Starting training...")
    epoch_train_loop.loop(
        get_data_stream=lambda: data_io.async(
            utterance_frame_data.batched_training_stream(),
            queue_size=5
        ),
        item_action=train,
        epoch_callback=validate_report
    )
