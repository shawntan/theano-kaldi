import theano
import theano.tensor as T
import ark_io
import numpy as np
import model
import data_io
import cPickle as pickle
import sys
import feedforward
from theano_toolkit.parameters import Parameters

if __name__ == "__main__":
    import config
    import utterance_vae
    import train_vae_features

    config.file("vae_model","Pickle file containing vae model.")
    config.parse_args()
    encoder = train_vae_features.feature_generator() 
    stream = data_io.context(ark_io.parse_binary(sys.stdin),left=5,right=5)
    count = 0
    for name,frames in stream:

        ark_io.print_ark_binary(sys.stdout,name,encoder(frames))
        count += 1
        if count % 100 == 0:
            print >> sys.stderr, "Generated %d utterances."%count
