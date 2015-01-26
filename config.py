import argparse
import theano.tensor as T
parser = argparse.ArgumentParser()

parser.add_argument(
		'--frames-file',
		dest = 'frames_file',
		required = True,
		type = str,
		help = ".pklgz file containing pickled (name,frames) pairs for training"
	)

parser.add_argument(
		'--labels-file',
		dest = 'labels_file',
		required = True,
		type = str,
		help = ".pklgz file containing pickled (name,frames) pairs for training"
	)

parser.add_argument(
		'--output-file',
		dest = 'output_file',
		required = True,
		type = str,
		help = ".pkl containing model parameters."
	)

parser.add_argument(
		'--structure',
		dest = 'structure',
		default = [360] + [1024] * 5 + [1874],
		type=lambda x:map(int,x.split(':')),
		help='Structure for network (default: "360:1024:1024:1024:1024:1024:1874")'
	)

parser.add_argument(
		'--tanh',
		dest = 'activation',
		action = 'store_const',
		const = T.tanh, default = T.nnet.sigmoid,
		help='Use tanh as activation function.'
	)

parser.add_argument(
		'--minibatch',
		dest = 'minibatch',
		default = 128,
		type = int,
		help = 'Size of minibatch'
	)

parser.add_argument(
		'--max-epochs',
		dest = 'max_epochs',
		default = 5,
		type = int,
		help = "Maximum number of epochs"
	)


frames_file = None
labels_file = None
output_file = None
hidden_activation = T.nnet.sigmoid
input_size = 360
layer_sizes = [1024]*5
output_size = 1874
minibatch = 128
max_epochs = 5
args = None
def parse_args():
	global hidden_activations,\
			input_size,\
			layer_sizes,\
			output_size,\
			frames_file,\
			labels_file,\
			output_file,\
			minibatch,\
			max_epochs,\
			args
	args = parser.parse_args()
	hidden_activation = args.activation
	input_size = args.structure[0]
	layer_sizes = args.structure[1:-1]
	output_size = args.structure[-1]
	frames_file = args.frames_file
	labels_file = args.labels_file
	output_file = args.output_file
	minibatch = args.minibatch
	max_epochs = args.max_epochs

