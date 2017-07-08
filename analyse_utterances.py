import theano.tensor as T
import theano
import sys, errno

from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import ark_io
import data_io
import config
import logging
import utterance_vae_shared
from theano_toolkit import utils as U
import utterance_frame_data

@config.option("model_file","Model file.",type=config.file)
def load_model_file(P,model_file):
#    logging.debug("Loading %s ..."%model_file)
    P.load(model_file)

def build_model(P,X,utt_lengths):
    mask = T.arange(X.shape[1]).dimshuffle('x',0) < \
                utt_lengths.dimshuffle(0,'x')

    speaker_encode, _ = utterance_vae_shared.build_encoder(P)
    _, utterance_speaker_mean, utterance_speaker_std = speaker_encode(X,mask)
    return utterance_speaker_mean[0], utterance_speaker_std[0]

if __name__ == "__main__":
    config.parse_args()
    P = Parameters()
    X = T.tensor3('X')
    utt_lengths = T.ivector('utt_lengths')
    f = theano.function(
            inputs=[X,utt_lengths],
            outputs=build_model(P,X,utt_lengths)
        )

    load_model_file(P)
    for x in utterance_frame_data.batched_training_stream():
        break
    frames, lengths = x
    print f(frames,lengths)
