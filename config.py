import sys
import argparse
import logging
import inspect
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
args = None

int = int
str = str
float = float


def file(filename):
    import os.path
    if filename != "":
        print >> sys.stderr, filename
        assert(os.path.isfile(filename))
    return filename

def structure(str_struct):
    return map(int, str_struct.split(':'))



def parse_args():
    global args
    parser.add_argument(
        '--log',
        dest="log",
        required=False,
        type=str,
        default="-",
        help="File for logging into."
    )
    args = parser.parse_args()
    if args.log == "-":
        log_fh = sys.stdout
    else:
        log_fh = open(args.log, 'w')
    logging.basicConfig(
            stream=log_fh,
            level=logging.DEBUG,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )



def option(var_name, description, type=str, default=None, nargs=None):
    arg_name = var_name.replace("_", "-")
    parser.add_argument(
        '--%s' % arg_name,
        dest=var_name,
        required=(default is None),
        type=type,
        default=default,
        help=description,
        nargs=nargs
    )
    def wrap(fun):
        arg_names = inspect.getargspec(fun)[0]

        def wrapped_fun(*fargs, **kwargs):
            arg_dict = kwargs
            if var_name not in arg_dict:
                arg_dict[var_name] = getattr(args, var_name)
            if len(arg_names) > 0:
                arg_dict.update(dict(zip(arg_names, fargs)))
                return fun(**arg_dict)
            else:
                return fun(*fargs, **arg_dict)
        return wrapped_fun
    return wrap
