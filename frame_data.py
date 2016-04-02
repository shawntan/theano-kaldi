import config
import data_io
from itertools import izip

@config.option("left_context","Number of frame contexts to the left.",
                type=config.int,default=5)
@config.option("right_context","Number of frame contexts to the right.",
                type=config.int,default=5)
def create_split_streams(frame_files,left_context,right_context):
    streams = []
    for frame_file in frame_files:
        stream = data_io.stream_file(frame_file)
        stream = data_io.context(stream,
                                 left=left_context,right=right_context)
        stream = data_io.zip_streams(stream)
        streams.append(stream)
    return streams

@config.option("training_frame_files","Files for training frames.",
                type=config.file,nargs='+')
def training_stream(training_frame_files):
    split_streams = create_split_streams(training_frame_files)
    split_streams = [ data_io.buffered_random(s) for s in split_streams ]
    split_streams = [ data_io.chop(s) for s in split_streams ]
    stream = data_io.random_select_stream(*split_streams)
    stream = data_io.buffered_random(stream)
    return stream

@config.option("validation_frame_files","Files for validation frames.",
                type=config.file,nargs='+')
def validation_stream(validation_frame_files):
    from itertools import chain
    split_streams = create_split_streams(validation_frame_files)
    stream = chain(*split_streams)
    return stream
