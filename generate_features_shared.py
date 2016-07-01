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

@config.option("left_context","Number of frame contexts to the left.",
                type=config.int,default=5)
@config.option("right_context","Number of frame contexts to the right.",
                type=config.int,default=5)
def input_stream(left_context,right_context):
    stream = ark_io.parse_binary(sys.stdin)
    stream = data_io.context(stream,left=left_context,right=right_context)
    return stream

@config.option("model_file","Model file.",type=config.file)
def load_model_file(P,model_file):
#    logging.debug("Loading %s ..."%model_file)
    P.load(model_file)

@config.option("sample","Sample from distribution?",type=config.int)
def build_model(P,X,sample):
    sample = bool(sample)
    X = X.dimshuffle('x',0,1)
    speaker_encode, acoustic_encode = utterance_vae_shared.build_encoder(P)
    mask = T.ones_like(X[:,:,0])
    _, utterance_speaker_mean,\
                utterance_speaker_std = speaker_encode(X,mask)
    if sample:
        print >> sys.stderr, "Sampling.", sample
        utterance_speaker = utterance_speaker_mean.dimshuffle(0,'x',1) +\
                            utterance_speaker_std.dimshuffle(0,'x',1) *\
                            U.theano_rng.normal(size=(
                                utterance_speaker_std.shape[0],
                                X.shape[1],
                                utterance_speaker_std.shape[1]
                            ))
    else:
        print >> sys.stderr, "No sampling."
        utterance_speaker = utterance_speaker_mean.dimshuffle(0,'x',1)
 

    acoustic, acoustic_mean, acoustic_std = acoustic_encode(X,utterance_speaker)

    if sample:
        output = T.concatenate([utterance_speaker[0],acoustic[0]], axis=1)
    else:
        acoustic = acoustic_mean
        output = T.concatenate([
                    T.tile(utterance_speaker[0,0],(acoustic[0].shape[0],1)),
                    acoustic[0]
                ], axis=1)

    return output

if __name__ == "__main__":
    config.parse_args()
    P = Parameters()
    X = T.matrix('X')
    output = build_model(P,X)

    transform = theano.function(
            inputs=[X],
            outputs=output
        )

    load_model_file(P)
    try:
        for name,frames in input_stream():
            ark_io.print_ark_binary(sys.stdout,name,transform(frames))
    except IOError as e:
        if e.errno == errno.EPIPE:
            print >> sys.stderr, "Stream stopped."
