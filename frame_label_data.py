import config
import data_io
from itertools import izip, chain


@config.option("left_context", "Number of frame contexts to the left.",
               type=config.int, default=5)
@config.option("right_context", "Number of frame contexts to the right.",
               type=config.int, default=5)
def create_split_streams(frame_files, label_files,
                         left_context, right_context):
    streams = []
    for frame_file, label_file in izip(frame_files, label_files):
        frame_stream = data_io.stream_file(frame_file)
        frame_stream = data_io.context(frame_stream,
                                       left=left_context, right=right_context)
        label_stream = data_io.stream_file(label_file)
        stream = data_io.zip_streams(frame_stream, label_stream)
        streams.append(stream)
    return streams


@config.option("training_frame_files", "Files for training frames.",
               type=config.file, nargs='+')
@config.option("training_label_files", "Files for training labels.",
               type=config.file, nargs='+')
def training_stream(training_frame_files, training_label_files):
    split_streams = create_split_streams(training_frame_files,
                                         training_label_files)
    split_streams = [data_io.buffered_random(s, buffer_items=20)
                     for s in split_streams]
    split_streams = [data_io.chop(s) for s in split_streams]
    stream = data_io.random_select_stream(*split_streams)
    stream = data_io.buffered_random(stream, buffer_items=100)
    return stream


@config.option("validation_frame_files", "Files for validation frames.",
               type=config.file, nargs='+', default='')
@config.option("validation_label_files", "Files for validation labels.",
               type=config.file, nargs='+', default='')
def validation_stream(validation_frame_files, validation_label_files):
    split_streams = create_split_streams(
        validation_frame_files,
        validation_label_files
    )
    stream = chain(*split_streams)
    return stream
