import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import feedforward
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters
def softmax_sample(lin_output):
    probs = T.nnet.softmax(lin_output)
    mask = U.theano_rng.multinomial(pvals=probs)
    score_ = mask * T.exp(lin_output)
    score = score_ / T.sum(score_,axis=-1).dimshuffle(0,'x')
    return probs,mask,score
    
def build_Y_Z2_generative(P,
        input_size,
        y_hidden_sizes,y_output_size,
        z2_hidden_sizes,z2_output_size,
        z1_decoder_sizes):

    infer_Y = feedforward.build_classifier(
            P,"infer_y",[input_size],y_hidden_sizes,y_output_size,
            output_activation=softmax_sample)
    infer_Z2 = vae.build_inferer(
            P,"infer_z2",[y_output_size,input_size],z2_hidden_sizes,z2_output_size,
            activation=T.nnet.softplus,
            initial_weights=feedforward.relu_init,
            initialise_outputs=True
            )
    generate_Z1 = vae.build_inferer(
            P,"infer_z1",[y_output_size,z2_output_size],z1_decoder_sizes,input_size,
            activation=T.nnet.softplus,
            initial_weights=feedforward.relu_init
            )
    
    def recon_error(Y,Z1,alpha=np.float32(0.1)):
        _,(Y_probs,_,_) = infer_Y([Z1])
        Z2_latent, Z2_mean, Z2_logvar = infer_Z2([Y,Z1])
        _, recon_Z1_mean, recon_Z1_logvar = generate_Z1([Y,Z2_latent])
        classification_error = -T.mean(T.log(Y_probs[T.arange(Y.shape[0]),Y]),axis=0)
        kl_divergence = T.mean(vae.kl_divergence(Z2_mean,Z2_logvar))
        log_prob_recon = T.mean(vae.gaussian_log(recon_Z1_mean,recon_Z1_logvar,Z1))
        cost = -(log_prob_recon - kl_divergence)
        loss = cost + alpha * classification_error
        return recon_Z1_mean, classification_error, kl_divergence, -log_prob_recon, loss

    def recon_error_unseen(Z1):
        _,(Y_probs,mask,Y) = infer_Y([Z1])
        entropy_Y = -T.mean(T.dot(Y,T.log(Y_probs).T))
        Z2_latent, Z2_mean, Z2_logvar = infer_Z2([Y,Z1])
        _, recon_Z1_mean, recon_Z1_logvar = generate_Z1([Y,Z2_latent])
        kl_divergence_Z2 = T.mean(vae.kl_divergence(Z2_mean,Z2_logvar))
        log_prob_recon   = T.mean(vae.gaussian_log(recon_Z1_mean,recon_Z1_logvar,Z1))
        cost = -(log_prob_recon - kl_divergence_Z2 + entropy_Y)
        loss = cost
        return recon_Z1_mean, kl_divergence_Z2, -log_prob_recon, loss, mask

    return infer_Y,infer_Z2,recon_error,recon_error_unseen



if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    recon_error = build_Y_Z2_generative(P,
        input_size=10,
        y_hidden_sizes=[5,5],y_output_size=20,
        z2_hidden_sizes=[10,10],z2_output_size=10,
        z1_decoder_sizes=[10,10])
    print recon_error(T.constant(np.random.randn(20,10)))[1].eval()




