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

args = None
def parse_args():
	global args
	args = parser.parse_args()
