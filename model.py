import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
import config
import vae
import feedforward
def build_unsupervised(P,input_size,layer_sizes,output_size):
    return vae.build(P, "z1_latent",
         input_size=input_size,
         encoder_hidden_sizes=layer_sizes,
         latent_size=output_size,
         activation=T.nnet.softplus
      )

def build_Y_Z2_generative(P,
        input_size,
        y_hidden_sizes,y_output_size,
        z2_hidden_sizes,z2_output_size,
        z1_decoder_sizes):
    infer_Y = feedforward.build_classifier(
            P,"infer_y",[input_size],y_hidden_sizes,y_output_size)
    infer_Z2 = vae.build_inferer(
            P,"infer_z2",[y_output_size,input_size],z2_hidden_sizes,z2_output_size)
    generate_Z1 = vae.build_inferer(
            P,"infer_z1",[y_output_size,z2_output_size],z1_decoder_sizes,input_size)
    
    def recon_error(Y,Z1,alpha=np.float32(0.1)):
        Z2_latent, Z2_mean, Z2_logvar = infer_Z2([Y,Z1])
        _, recon_Z1_mean, recon_Z1_logvar = generate_Z1([Y,Z2_latent])
        Y_probs = infer_Y([Z1])
        classification_error = -T.log(Y_probs[T.arange(Y.shape[0]),Y])
        cost = -(-vae.kl_divergence(Z2_mean,Z2_logvar) +\
                    vae.gaussian_log(recon_Z1_mean,recon_Z1_logvar,Z1))
        loss = T.mean(cost) + alpha * T.mean(classification_error)
        return recon_Z1_mean, T.mean(classification_error,axis=0), loss
    def recon_error_unseen(Z1):
        Y = T.arange(y_output_size).dimshuffle('x',0)     # 1 x y_output_size
        Z1_ = Z1.dimshuffle(0,'x',1)                      # batch_size x 1 x input_size
        Z2_latent, Z2_mean, Z2_logvar = infer_Z2([Y,Z1_]) # batch_size x y_output_size x z2_output_size
        _, recon_Z1_mean, recon_Z1_logvar = generate_Z1([Y,Z2_latent])
                                                          # batch_size x y_output_size x input_size
        Y_probs = infer_Y([Z1]) # batch_size x y_output_size
        L = -(vae.gaussian_log(recon_Z1_mean,recon_Z1_logvar,Z1_) \
                - vae.kl_divergence(Z2_mean,Z2_logvar))
                                # batch_size x y_output_size
        neg_log_Y_probs = -T.log(Y_probs) #-L + 
        cost = -T.dot(Y_probs,(-L + neg_log_Y_probs).T)
        return recon_Z1_mean, T.mean(cost)
    return infer_Y,infer_Z2,recon_error,recon_error_unseen



if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    recon_error = build_Y_Z2_generative(P,
        input_size=10,
        y_hidden_sizes=[5,5],y_output_size=20,
        z2_hidden_sizes=[10,10],z2_output_size=10,
        z1_decoder_sizes=[10,10])
    print recon_error(
#            T.arange(20),
            T.constant(np.random.randn(20,10))
        )[1].eval()




