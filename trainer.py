import json
import config
import numpy as np
import data_io
from itertools import izip
import theano.tensor as T
import theano
import math
from theano_toolkit import updates

def data_stream(file_sequences):
    split_streams = [ data_io.stream(*files) for files in izip(*file_sequences) ]
    stream = data_io.random_select_stream(*split_streams)
    stream = data_io.buffered_random(stream)
    stream = data_io.randomise(stream)
    return stream

def define_shared_variable(data_variable):
    shared_var = theano.shared(np.zeros((1,) * data_variable.ndim,
                                               dtype=data_variable.dtype)) 
    shared_var.name = "%s_shared"%data_variable
    return shared_var

def build_train_epoch(data_variables):
    for var in data_variables:
        config.file_sequence("%s_files"%var,".pklgz file containing %s."%var)
    config.integer("minibatch","Minibatch size.",default=512)
    learning_curve = open("learning_curve",'w',buffering=1)
    def compile_train_epoch(parameters,gradients,update_vars,outputs=None,data_stream=data_stream,
            update_strategy=updates.momentum):
        file_sequences = [ getattr(config.args,"%s_files"%var) for var in data_variables ]
        assert(all(len(file_sequences[0]) == len(file_sequences[i]) for i in xrange(len(file_sequences))))
        shared_variables = [ define_shared_variable(var) for var in data_variables ]
        start_idx = T.iscalar('start_idx')
        end_idx = T.iscalar('end_idx')
        lr = T.scalar('lr')
        train = theano.function(
                inputs  = [lr,start_idx,end_idx],
                outputs = outputs,
                updates = update_strategy(parameters,gradients,learning_rate=lr,P=update_vars),
                givens  = { var:shared_var[start_idx:end_idx]
                                for var,shared_var in izip(data_variables,shared_variables) },
            )
        minibatch_size = config.args.minibatch
        def run_train(learning_rate):
            stream = data_stream(file_sequences)
            total_frames = 0
            for item in stream:
                size = item[-1]
                total_frames += item[0].shape[0]
                batch_count = int(math.floor(size/float(minibatch_size)))
                for shared_var,data in izip(shared_variables,item[:-1]):
                    shared_var.set_value(data)

                for idx in xrange(batch_count):
                    start = idx*minibatch_size
                    end = min((idx+1)*minibatch_size,size)
                    if outputs is not None:
                        print >> learning_curve, train(learning_rate,start,end)
                    else:
                        train(learning_rate,start,end)
        return run_train
    return compile_train_epoch



def build_train_loop():
    config.file("output_file","Output file.")
    config.file("temporary_file","Temporary file.")
    config.file("learning_file","Temporary file.")
    config.integer("max_epochs","Maximum number of epochs to train.",default=200)
    config.real("improvement_threshold","Ratio of cost reduction or learning rate will be multiplied by learning_rate_decay",default=0.95)
    config.real("learning_rate","Learning rate to start at")
    config.real("learning_rate_decay","Proportion to reduce learning_rate")
    config.real("learning_rate_minimum","Learning rate value to stop at")

    def train_loop(logging,run_test,run_train,P,update_vars,monitor_score='cross_entropy'):
        starting_learning_rate = config.args.learning_rate
        max_epochs = config.args.max_epochs
        temporary_model_file = config.args.temporary_file
        update_parameters_file = config.args.learning_file
        output_file = config.args.output_file
        improvement_threshold = config.args.improvement_threshold
        learning_rate = config.args.learning_rate
        learning_rate_decay = config.args.learning_rate_decay
        learning_rate_minimum = config.args.learning_rate_minimum

        best_score = np.inf
        for epoch in xrange(max_epochs):
            scores = run_test()
            score = scores[monitor_score]
            logging.info("Epoch " + str(epoch) + " results: " + json.dumps(scores))
            _best_score = best_score

            if score < _best_score:
                logging.debug("score < best_score, saving model.")
                best_score = score
                P.save(temporary_model_file)
                update_vars.save(update_parameters_file)

            if (_best_score - score)/_best_score < (1 - improvement_threshold) and epoch > 0:
                learning_rate *= learning_rate_decay
                logging.debug("Halving learning rate. learning_rate = " + str(learning_rate))

                if score > _best_score:
                    logging.debug("Loading previous model.")
                    P.load(temporary_model_file)
                    update_vars.load(update_parameters_file)

            if learning_rate < learning_rate_minimum: break

            logging.info("Epoch %d training."%(epoch + 1))
            run_train(learning_rate)
            logging.info("Epoch %d training done."%(epoch + 1))

        P.load(temporary_model_file)
        P.save(output_file)
        logging.debug("Done training process.")
    return train_loop
