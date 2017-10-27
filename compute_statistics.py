import theano
import theano.tensor as T
from theano_toolkit.parameters import Parameters

import config
import model
import frame_label_data
import logging
import numpy as np


@config.option("model_file", "Saved parameters.")
def load_state(P, model_file):
    logging.info("Loading previous model and state.")
    P.load(model_file, strict=False)


@config.option("augmented_file", "Augmented with stats")
def save_state(P, augmented_file):
    logging.info("Saving new model.")
    P.save(augmented_file)


if __name__ == "__main__":
    config.parse_args()
    X = T.matrix('X')
    new_stats = {}
    while True:
        P = Parameters()
        for stat_name in new_stats:
            print "Adding ", stat_name
            P[stat_name] = new_stats[stat_name]
        predict = model.build(P)
        _, outputs = predict(X)
        load_state(P)
        for stat_name in new_stats:
            P[stat_name].set_value(new_stats[stat_name].astype(np.float32))

        var_sum, var_sqr_sum, var_name = None, None, None
        for n in theano.gof.graph.io_toposort([X], [outputs]):
            if hasattr(n.out, 'name') and n.out.name is not None:
                if n.out.name.endswith('_bn_statistic'):
                    var_sum = T.sum(n.out, axis=0)
                    var_sqr_sum = T.sum(T.sqr(n.out), axis=0)
                    var_name = n.out.name
                    break
        if var_name is None:
            break
        print var_name
        stats = theano.function(
            inputs=[X],
            outputs=[var_sum, var_sqr_sum]
        )
        total_count = 0
        total_stat_sum = 0
        total_stat_sqr_sum = 0
        for x, _ in frame_label_data.training_stream():
            stat_sum, stat_sqr_sum = stats(x)
            total_count += x.shape[0]
            total_stat_sum += stat_sum
            total_stat_sqr_sum += stat_sqr_sum

        mean = total_stat_sum / total_count
        sqr_mean = total_stat_sqr_sum / total_count
        var = sqr_mean - mean**2
        print mean
        print var
        print "Assigning %s statistics to parameters.." % var_name
        new_stats["%s_mean" % var_name] = mean
        new_stats["%s_var" % var_name] = var
    P = Parameters()
    predict = model.build(P)
    for stat_name in new_stats:
        print "Adding ", stat_name
        P[stat_name] = new_stats[stat_name]
    _, outputs = predict(X)
    load_state(P)
    save_state(P)
