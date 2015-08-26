import sys
import argparse
import logging
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

file("log","File for logs.",default="-")
args = None
def parse_args():
	global args
	args = parser.parse_args()
	if args.log == "-":
		log_fh = sys.stdout
	else:
		log_fh = open(args.log,'w')
		print "Logging to " + args.log
	logging.basicConfig(
			stream=log_fh,
			level=logging.DEBUG,
			format="%(asctime)s:%(levelname)s:%(message)s"
		)

