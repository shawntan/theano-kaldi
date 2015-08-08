
import config
if __name__ == "__main__":
	config.parser.description = "theano-kaldi script for pretraining models using stacked denoising autoencoders."
	config.file_sequence("frames_files",".pklgz file containing audio frames.")
	config.structure("generative_structure","Structure of generative model.")
	config.file("validation_frames_file","Validation set.")
	config.file("output_file","Output file.")
	config.file("spk2utt_file","spk2utt file from Kaldi.")
	config.integer("minibatch","Minibatch size.",default=128)
	config.integer("max_epochs","Maximum number of epochs to train.",default=20)
	config.parse_args()
	
import theano
import theano.tensor as T
import theano_toolkit 
import numpy as np
import math
import sys
import random

import data_io
import model
import updates
import cPickle as pickle
from itertools import izip, chain


import theano_toolkit.utils   as U
from theano_toolkit.parameters import Parameters
import vae_sa

from pprint import pprint

def get_speaker_ids(spk2utt_file):
	return { lines.split()[0]:idx
			 for idx,lines in enumerate(open(spk2utt_file)) }

def frame_speaker_stream(stream,speaker_ids):
	for tup in stream:
		name = tup[0]
		frames = tup[1]
		speaker = name.split('_')[0]
		ids = np.empty(frames.shape[0],dtype=np.int32)
		ids.fill(speaker_ids[speaker])
		yield tup[1:] + (ids,)

def speaker_grouped_stream(frames_files):
	streams = [ data_io.stream(f,with_name=True) for f in frames_files ]
	stream_next = [ s.next() for s in streams ]

	frames_buf = speakers_buf = None
	frame_count = 0
	while len(streams) > 0:
		stream_idx = random.randint(0,len(streams)-1)
		try:
			group = []

			name,frames = stream_next[stream_idx]
			batch_speaker = speaker = name.split('_')[0]
			while speaker == batch_speaker:
				group.append((name,frames))

				stream_next[stream_idx] = streams[stream_idx].next()

				name,frames = stream_next[stream_idx]
				speaker = name.split('_')[0]
		except StopIteration:
			streams = streams[:stream_idx] + streams[stream_idx+1:]
			stream_next = stream_next[:stream_idx] + stream_next[stream_idx+1:]
		if len(group) > 0: yield group

def randomised_speaker_groups(grouped_stream,speaker_ids,
		buffer_size=2**17,
		validation_set=None,
		validation_utt_count=1):
	if validation_set == None:
		validation_set = config.args.validation_frames_file
	import gzip,os
	frames_buf = None
	speakers_buf = None

	frame_count = 0

	if not os.path.isfile(validation_set):
		validation_file = gzip.open(validation_set,'w')
	else:
		validation_file = None
	
	for group in grouped_stream:
		speaker = group[0][0].split('_')[0]
		group_start_frame = frame_count
		if validation_file: pickle.dump(group[0],validation_file,2)
		for name,frames in group[1:]:
			if frames_buf is None:
				frames_buf = np.empty((buffer_size,frames.shape[1]),dtype=np.float32)
				speakers_buf = np.empty((buffer_size,),dtype=np.int32)

			if frame_count + frames.shape[0] > buffer_size:
				if frame_count > group_start_frame:
#					print "(shuffle %d,%d)"%(group_start_frame,frame_count),
					np.random.shuffle(frames_buf[group_start_frame:frame_count])
#				print "yield"
				yield frames_buf,speakers_buf,frame_count
				group_start_frame = frame_count = 0

#			print speaker,
#			frames[:,-1] = speaker_ids[speaker]
			frames_buf[frame_count:frame_count+frames.shape[0]] = frames
			speakers_buf[frame_count:frame_count+frames.shape[0]] = speaker_ids[speaker]
			frame_count = frame_count + frames.shape[0]

		if frame_count > group_start_frame:
#			print "(shuffle %d,%d)"%(group_start_frame,frame_count),
			np.random.shuffle(frames_buf[group_start_frame:frame_count])

	
	yield frames_buf,speakers_buf,frame_count
	if validation_file: validation_file.close()
	

if __name__ == "__main__":
	frames_files = config.args.frames_files
	
	minibatch_size = config.args.minibatch

	print config.args.generative_structure
	input_size  = config.args.generative_structure[0]
	layer_sizes = config.args.generative_structure[1:-1]
	output_size = config.args.generative_structure[-1]
	speaker_ids = get_speaker_ids(config.args.spk2utt_file)
	X = T.matrix('X')
	S = T.ivector('S')
	start_idx = T.iscalar('start_idx')
	end_idx = T.iscalar('end_idx')
	X_shared = theano.shared(np.zeros((1,input_size),dtype=theano.config.floatX))
	S_shared = theano.shared(np.zeros((1,),dtype=np.int32))

	def test_validation(test):
		total_errors = 0
		total_frames = 0
		for f,s in frame_speaker_stream(
				data_io.stream(config.args.validation_frames_file,with_name=True),speaker_ids):
			errors = np.array(test(f,s))
			total_frames += f.shape[0]
			total_errors += f.shape[0] * errors
		return total_errors/total_frames
	
	def train_epoch(train):
		stream = randomised_speaker_groups(
				speaker_grouped_stream(frames_files),
				speaker_ids
			)
		total_count = 0
		total_loss  = 0
		for f,s,size in stream:
			X_shared.set_value(f)
			S_shared.set_value(s)
			batch_count = int(math.ceil(size/float(minibatch_size)))
			seq = range(batch_count)
			random.shuffle(seq)
			for idx in seq:
				start = idx*minibatch_size
				end = min((idx+1)*minibatch_size,size)
#				print f[start:end,-1]
#				print s[start:end]
#				print f[start:end,-1] == s[start:end]
				loss = train(start,end)

	prev_P = None
	train = None
	for layer in xrange(len(layer_sizes)):
		P = Parameters()
		_, recon_error = vae_sa.build(P, "vae",
					input_size,
					layer_sizes[:layer+1],
					output_size,
					speaker_count = len(speaker_ids),
					speaker_embedding_size = 100,
					activation=T.nnet.sigmoid
				)
	
		if layer > 0:
			print "decoder_output to decoder_output"
			P.W_vae_decoder_output.set_value(prev_P.W_vae_decoder_output.get_value())
			P.b_vae_decoder_output.set_value(prev_P.b_vae_decoder_output.get_value())
			for i in xrange(layer):
				print "encoder_%d to encoder_%d"%(i,i)
				P["W_vae_encoder_hidden_%d"%i].set_value(
						prev_P["W_vae_encoder_hidden_%d"%i].get_value())
				P["b_vae_encoder_hidden_%d"%i].set_value(
						prev_P["b_vae_encoder_hidden_%d"%i].get_value())
				if i > 0:
					print "decoder_%d to decoder_%d"%(i,i+1)
					P["W_vae_decoder_hidden_%d"%(i+1)].set_value(
							prev_P["W_vae_decoder_hidden_%d"%i].get_value())
					P["b_vae_decoder_hidden_%d"%(i+1)].set_value(
							prev_P["b_vae_decoder_hidden_%d"%i].get_value())


		parameters = P.values()
		
		X_recon,cost = recon_error(X,S)
		loss = cost + 0.5 * sum(T.sum(w**2) for w in parameters if "embedding" not in w.name)
		gradients  = T.grad(cost,wrt=parameters)
		pprint(sorted((p.name,p.get_value().shape) for p in parameters ))
		print "Compiling function...",
		train = theano.function(
				inputs = [start_idx,end_idx],
				updates = updates.adadelta(parameters,gradients,eps=1e-8),
				givens  = {
					X: X_shared[start_idx:end_idx],
					S: S_shared[start_idx:end_idx],
				}
			)
		test = theano.function(
				inputs = [X,S],
				outputs = [T.mean(T.sum((X-X_recon)**2,axis=1)),cost],
			)
		print "Done."
		best_score = np.inf
		for _ in xrange(config.args.max_epochs):
			train_epoch(train)
			scores = test_validation(test)
			print scores,
			score = scores[-1]
			if score < best_score:
				best_score = score
				P.save(config.args.output_file)
				print "Saved."
			else:
				print
		prev_P = P

