import argparse
import theano.tensor as T
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def file_sequence(var_name,description="",default=None):
	arg_name = var_name.replace("_","-")
	parser.add_argument(
			'--%s' % arg_name,
			nargs = '+',
			dest = var_name,
			required = (default == None),
			type = str,
			default = default,
			help = description
		)
def file(var_name,description="",default=None):
	arg_name = var_name.replace("_","-")
	parser.add_argument(
			'--%s' % arg_name,
			dest = var_name,
			required = (default == None),
			type = str,
			default = default,
			help = description
		)

def structure(var_name,description="",default=None):
	arg_name = var_name.replace("_","-")
	parser.add_argument(
			'--%s' % arg_name,
			dest = var_name, 
			required = (default == None),
			type=lambda x:map(int,x.split(':')),
			default = default,
			help=description
		)

def integer(var_name,description="",default=None):
	arg_name = var_name.replace("_","-")
	parser.add_argument(
			'--%s' % arg_name,
			dest = var_name,
			required = (default == None),
			default = default,
			type = int,
			help =description 
		)


#parser.add_argument(
#		'--tanh',
#		dest = 'activation',
#		action = 'store_const',
#		const = T.tanh, default = T.nnet.sigmoid,
#		help='Use tanh as activation function.'
#	)


frames_files = None
labels_files = None
output_files = None
hidden_activation = T.nnet.sigmoid
input_size = 360
layer_sizes = [1024]*5
output_size = 1874
minibatch = 128
max_epochs = 5
args = None
def parse_args():
	global args
	args = parser.parse_args()
