import config
import theano.tensor as T
import feedforward


@config.option("structure", "Structure of the descriminative model.",
               type=config.structure)
@config.option("weights_file", "Loading weights for the model.",
               type=config.file, default="")
def build(P, structure, weights_file, training=True):
    input_size = structure[0]
    layer_sizes = structure[1:-1]
    output_size = structure[-1]

    classifier = feedforward.build_classifier(
        P, "classifier",
        [input_size], layer_sizes, output_size,
        activation=T.nnet.sigmoid,
        initial_weights=feedforward.initial_weights,
        batch_norm=True
    )

    def predict(X):
        hiddens, outputs = classifier([X])
        return hiddens, outputs
    if weights_file != "":
        P.load(weights_file)
    return predict
