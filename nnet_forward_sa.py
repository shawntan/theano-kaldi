import config
config.structure("generative_structure","Structure of generative model.")
config.structure("discriminative_structure","Structure of discriminative model.")
config.file("generative_model",".pkl file containing generative model.")
config.file("discriminative_model",".pkl file containing discriminative model.")
config.file("class_counts",".counts file giving counts of all the pdfs.")
config.file("spk2utt_file","spk2utt file from Kaldi.")
config.integer("sample","Number of times to sample from z.",default=1)
config.parse_args()
import cPickle as pickle
import theano
import theano.tensor as T
import ark_io
import numpy as np
import model

import sys
import vae
import feedforward
from theano_toolkit.parameters import Parameters
from theano_toolkit import utils as U
from train_sa_vae import *
def ark_stream():
	return ark_io.parse(sys.stdin)

def create_model(
		gen_filename,gen_structure,
		dis_filename,dis_structure,
		speaker_ids,
		counts,sample=1):

	X = T.matrix('X')
	S = T.ivector('S')
	
	P_vae  = Parameters()
	P_disc = Parameters()
	speaker_count = len(speaker_ids)
	speaker_embedding_size = 100

	sample_encode, recon_error = vae_sa.build(P_vae, "vae",
				gen_structure[0],
				gen_structure[1:-1],
				gen_structure[-1],
				speaker_count = speaker_count,
				speaker_embedding_size = speaker_embedding_size,
				activation=T.nnet.sigmoid
			)
	discriminate = feedforward.build(
		P_disc,
		name = "discriminate",
		input_size   = dis_structure[0], 
		hidden_sizes = dis_structure[1:-1],
		output_size  = dis_structure[-1],
		activation=T.nnet.sigmoid
	)
	
	log_output_total = 0
	mean, logvar, _ = sample_encode(X,S)
	noise = U.theano_rng.normal(size=(logvar.shape[0] * sample, logvar.shape[1]))
	for i in xrange(sample):
		e = noise[i * logvar.shape[0]:(i + 1) * logvar.shape[0]]
		latent = mean + e * T.exp(0.5 * logvar)
		lin_output = discriminate(latent)
		log_output_total += T.log(T.nnet.softmax(lin_output))
	log_output = log_output_total / sample


	f = theano.function(
			inputs = [X,S],
			outputs = log_output - T.log(counts/T.sum(counts))
		)
	P_vae.load(gen_filename)
	P_disc.load(dis_filename)
	return f

def print_ark(name,array):
	print name,"["
	for i,row in enumerate(array):
		print " ",
		for cell in row:
			print "%0.6f"%cell,
		if i == array.shape[0]-1:
			print "]"
		else:
			print
	
if __name__ == "__main__":
	with open(config.args.class_counts) as f:
		row = f.next().strip().strip('[]').strip()
		counts = np.array([ np.float32(v) for v in row.split() ])
	speaker_ids = get_speaker_ids(config.args.spk2utt_file)
	predict = create_model(
		gen_filename=config.args.generative_model,
		gen_structure=config.args.generative_structure,
		dis_filename=config.args.discriminative_model,
		dis_structure=config.args.discriminative_structure,
		speaker_ids = speaker_ids,
		counts=counts
	)

	if predict != None:
		for name,frames in ark_stream():
			speaker = name.split('_')[0]
			
			log_probs = sum(
					predict(
						frames,
						speaker_ids[speaker] * np.ones((frames.shape[0],),dtype=np.int32)
					) for _ in xrange(config.args.sample)
				)/config.args.sample
			print_ark(name,log_probs)
