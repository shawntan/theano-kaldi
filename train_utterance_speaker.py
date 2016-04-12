import sys
import logging,json
import theano
import theano.tensor as T
import numpy as np
import math
import cPickle as pickle
from pprint import pprint
from itertools import izip, chain, tee
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

if __name__ == "__main__":
    import config
    config.parser.description = "theano-kaldi script for fine-tuning DNN feed-forward models."
    config.file_sequence("frames_files","Frames file.")
    config.file_sequence("validation_frames_files","Validation set frames file.")
    config.file("pooling_method","Method for pooling across utterance")
    config.file("utt2spk_file","Utterance to speaker mapping.")
    config.file("validation_utt2spk_file","Utterance to speaker mapping.")
    config.file("output_file","File to write the generative model to")
    config.file("learning_curve","File to write the costs")
    config.integer("epochs","Number of epochs")
    config.integer("batch_size","Batch size.")
    config.integer("gradient_clip","Gradient clip size")
    config.real("learning_rate","Learning rate.")
    config.parse_args()

import utterance_vae
import data_io

def make_split_stream(frames_files,utt2spk_file):
    mapping = {}
    spk_ids = {}
    for line in open(utt2spk_file,'r'):
        utt_id,spk_id = line.strip().split()
        if spk_id not in spk_ids:
            idx = spk_ids[spk_id] = len(spk_ids)
        else:
            idx = spk_ids[spk_id]
        mapping[utt_id] = idx

    streams = []
    for frames_file in frames_files:
        stream1, stream2 = tee(data_io.stream_file(frames_file))
        frame_stream = data_io.context(stream1,left=5,right=5)
        spkr_stream = ((n,mapping[n]) for n,_ in stream2)
        stream = data_io.zip_streams(frame_stream,spkr_stream)
        streams.append(stream)
    return streams

def utterance_batch(stream,batch_size=10):
    def construct_batch(batch_buffer):
        min_frames = min(f.shape[0] for f,_ in batch_buffer)
        batch_frames = np.empty(
                (len(batch_buffer),min_frames,frames.shape[1]),dtype=np.float32)
        batch_speaker_id = np.array([s for _,s in batch_buffer],dtype=np.int32)
        for i,(f,s) in enumerate(batch_buffer):
            if f.shape[0] > min_frames:
                np.random.shuffle(f)
                batch_frames[i] = f[:min_frames]
            else:
                batch_frames[i] = f
        return batch_frames, batch_speaker_id

    batch_ptr = 0
    batch_buffer = batch_size * [None]

    for frames, speaker_id in stream:
        batch_buffer[batch_ptr] = frames, speaker_id
        batch_ptr += 1

        if batch_ptr == batch_size:
            yield construct_batch(batch_buffer)
            batch_ptr = 0

    if batch_ptr > 0:
        batch_buffer = batch_buffer[:batch_ptr]
        yield construct_batch(batch_buffer)


def batched_sort(stream,buffer_size=100):
    batch = buffer_size * [None]
    batch_ptr = 0
    for x in stream:
        batch[batch_ptr] = x
        batch_ptr += 1
        if batch_ptr == buffer_size:
            batch.sort(key=lambda x:x[0].shape[0])
            for x in batch: yield x
            batch_ptr = 0

    batch = batch[:batch_ptr]
    batch.sort(key=lambda x:x[0].shape[0])
    for x in batch: yield x

def build_data_stream(frames_files,utt2spk_file,context=5):
    streams = [ data_io.buffered_random(s,buffer_items=200)
                    for s in make_split_stream(frames_files,utt2spk_file) ]
    stream = data_io.random_select_stream(*streams)
    stream = data_io.buffered_random(stream,buffer_items=100)
    stream = batched_sort(stream)
    stream = utterance_batch(stream,batch_size=config.args.batch_size)
    stream = data_io.buffered_random(stream,buffer_items=200)
    stream = data_io.async(stream)
    return stream

def count_frames(labels_files,output_size):
    split_streams = [ data_io.stream(f) for f in labels_files ]
    frame_count = 0
    label_count = np.zeros(output_size,dtype=np.float32)
    for l in chain(*split_streams):
        frame_count += l.shape[0]
        np.add.at(label_count,l,1)
    return frame_count,label_count


def build_run_test(stream,outputs,**kwargs):
    keys = outputs.keys()
    test = theano.function(
            outputs=[outputs[k] for k in keys],
            **kwargs
        )
    report = { k: None for k in keys }
    def run_test():
        total = sum(np.array(test(*x)) for x in stream())
        for i in xrange(len(keys)): 
            report[keys[i]] = total[i]
        return report
    return run_test

if __name__ == "__main__":
    utterance_count = sum(
            1 for _ in chain(*make_split_stream(
                config.args.frames_files,
                config.args.utt2spk_file
            )))
#    batched_utterance_count = sum(x[0].shape[0] for x in build_data_stream(
#            config.args.frames_files,
#            config.args.utt2spk_file
#        ))
#    print utterance_count, batched_utterance_count
    logging.info("Training utterances count: %d"%utterance_count)
    P = Parameters()
    unsupervised_training_costs,\
        supervised_training_costs = utterance_vae.build(P,pooling_method=config.args.pooling_method)
    X = T.tensor3('X')
    speaker_id = T.ivector('spkr_id')

    speaker_latent_cost, acoustic_latent_cost, recon_cost = \
            unsupervised_training_costs(X)
    
    parameters = [ w for w in P.values() if w.name != 'speaker_vector']
    logging.info("Tuning parameters: " + ", ".join(w.name for w in parameters))
    cost = speaker_latent_cost + acoustic_latent_cost + recon_cost
    loss = cost +\
            (0.5/utterance_count) * sum(
                    T.sum(T.sqr(w)) for w in parameters 
                    if w.name not in \
	            ('speaker_vector','b_decode_mean','b_decode_std')
                )
            #sum(supervised_training_costs(X,speaker_id)) +\
    loss = loss / T.cast(X.shape[1],'float32')
    P_learning = Parameters()
    logging.debug('Calculating gradients...')
    gradients = T.grad(loss,wrt=parameters)
    gradients = updates.clip(config.args.gradient_clip)(parameters,gradients)
    logging.debug('Compiling...')
    train = theano.function(
            inputs=[X,speaker_id],
            outputs=[
                speaker_latent_cost,
                acoustic_latent_cost / X.shape[1],
                recon_cost / X.shape[1],
            ],
            updates=updates.adam(parameters,gradients,
				learning_rate=config.args.learning_rate,
                P=P_learning),
            on_unused_input='warn'
        )

    stream = chain(*make_split_stream(
            config.args.validation_frames_files,
            config.args.validation_utt2spk_file
        ))
    validation_set = list((x,) for x,_  in utterance_batch(stream,batch_size=1))
    run_test = build_run_test(
            stream=lambda: iter(validation_set),
            inputs=[X],
            outputs={
                "speaker_latent_cost":speaker_latent_cost,
                "acoustic_latent_cost":acoustic_latent_cost,
                "recon_cost":recon_cost
            },
            on_unused_input='warn'
        )

    learning_curve = open(config.args.learning_curve,'w',0)
    best_cost = np.inf


    for _ in xrange(config.args.epochs):

        report = run_test()
        logging.info(report)
        curr_cost = sum(v for v in report.values())
        logging.info("Current cost: %0.2f"%curr_cost)

        if curr_cost < best_cost:
            logging.info("Saving model. Best cost was %0.2f."%best_cost)
            best_cost = curr_cost
            P.save(config.args.output_file)
            P_learning.save(config.args.output_file + ".learning")
    
        for batch_frames,batch_speaker_id in build_data_stream(
                config.args.frames_files,
                config.args.utt2spk_file):
            #print batch_speaker_id, batch_frames.shape[1]
            values = train(batch_frames,batch_speaker_id)
            print >> learning_curve, '\t'.join(str(v) for v in values)
