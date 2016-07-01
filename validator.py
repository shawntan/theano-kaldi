import config
import logging
import numpy as np
import theano
def build(inputs,outputs,monitored_var,validation_stream,
        best_score_init=np.inf,
        callback=lambda best_score,current_score: None):

    output_keys = outputs.keys()
    test = theano.function(
            inputs=inputs,
            outputs=[outputs[k] for k in output_keys],
        )

    class Validator:
        def __init__(self):
            self.best_score = best_score_init
        
        def __call__(self):
            total_instances = 0
            total = sum(np.array(test(*x)) for x in validation_stream())
            report = { output_keys[i]: total[i] 
                            for i in xrange(len(output_keys)) }
            score = report[monitored_var]
            callback(self.best_score,score)
            if score < self.best_score:
                self.best_score = score
            return report

    return Validator()
