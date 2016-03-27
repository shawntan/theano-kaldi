import config
import data_io
from itertools import izip

@config.option("left_context","Number of frame contexts to the left.",
                type=config.int,default=5)
@config.option("right_context","Number of frame contexts to the right.",
                type=config.int,default=5)
def create_split_streams(frame_files,label_files,left_context,right_context):
    return [ data_io.zip_streams(
                data_io.context(
                    data_io.stream_file(frame_file),
                    left=left_context,right=right_context
                ),
                data_io.stream_file(label_file)
            ) for frame_file,label_file in izip(frame_files,label_files) ]



@config.option("training_frame_files","Files for training frames.",
                type=config.file,nargs='+')
@config.option("training_label_files","Files for training labels.",
                type=config.file,nargs='+')
def stream(training_frame_files,training_label_files):
    split_streams = create_split_streams(training_frame_files,training_label_files)
    stream = data_io.random_select_stream(*split_streams)
    stream = data_io.buffered_random(stream)
    return stream

@config.option("validation_frame_files","Files for validation frames.",
                type=config.file,nargs='+')
@config.option("validation_label_files","Files for validation frames.",
                type=config.file,nargs='+')
def stream(validation_frame_files,validation_label_files):
    from itertools import chain
    split_streams = create_split_streams(validation_frame_files,validation_label_files)
    stream = chain(*split_streams)
    return stream





