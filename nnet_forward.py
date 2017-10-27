import sys
import theano
import theano.tensor as T
import ark_io
import data_io
import numpy as np
from theano_toolkit.parameters import Parameters

import config
import model
import cPickle as pickle


model_file = config.option("model_file", "Saved parameters.")


@model_file
def first_load(P, model_file):
    data = pickle.load(open(model_file))
    for k in data:
        P[k] = data[k]


@model_file
def load(P, model_file):
    P.load(model_file, strict=False)


@config.option("class_counts_file", "Files for counts of each class.",
               type=config.file)
def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = f.next().strip().strip('[]').strip()
        counts = np.array([np.float32(v) for v in row.split()])
    return counts


def log_softmax(output):
    if output.owner.op == T.nnet.softmax_op:
        x = output.owner.inputs[0]
        k = T.max(x, axis=1, keepdims=True)
        sum_x = T.log(T.sum(T.exp(x - k), axis=1, keepdims=True)) + k
        print >> sys.stderr, "Stable log softmax"
        return x - sum_x
    else:
        return T.log(T.nnet.softmax(output))


if __name__ == "__main__":
    config.parse_args()

    X = T.matrix('X')
    P = Parameters(allow_overrides=True)
    first_load(P)
    predict = model.build(P)
    _, outputs = predict(X)
    counts = load_counts()
    predict = theano.function(
        inputs=[X],
        outputs=log_softmax(outputs) - T.log(counts / T.sum(counts))
    )
    load(P)
    buffer_size = 64
    if predict is not None:
        stream = data_io.context(ark_io.parse_binary(sys.stdin),
                                 left=5, right=5)
        stream_buffer = [None] * buffer_size
        ptr = 0
        for name, frames in stream:
            stream_buffer[ptr] = name, frames
            ptr += 1
            if ptr == buffer_size:
                prediction = predict(
                    np.concatenate([f for _, f in stream_buffer], axis=0)
                )
                ptr = 0

                out_ptr = 0
                for i, (b_name, b_frames) in enumerate(stream_buffer):
                    ark_io.print_ark_binary(
                        sys.stdout, b_name,
                        prediction[out_ptr:out_ptr + b_frames.shape[0]]
                    )
                    out_ptr = out_ptr + b_frames.shape[0]
        if ptr > 0:
            stream_buffer = stream_buffer[:ptr]
            prediction = predict(
                np.concatenate([f for _, f in stream_buffer], axis=0)
            )
            ptr = 0

            out_ptr = 0
            for i, (b_name, b_frames) in enumerate(stream_buffer):
                ark_io.print_ark_binary(
                    sys.stdout, b_name,
                    prediction[out_ptr:out_ptr + b_frames.shape[0]]
                )
                out_ptr = out_ptr + b_frames.shape[0]
