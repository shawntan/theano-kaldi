import config
import data_io
from itertools import izip
import numpy as np

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

def batched_utterances(stream,batch_size):
    def construct_batch(batch_buffer):
        batch_utt_lengths = np.array([f.shape[0] for (f,) in batch_buffer],dtype=np.int32)
        batch_frames = np.empty((
                len(batch_buffer),
                np.max(batch_utt_lengths),
                batch_buffer[0][0].shape[1]
            ),dtype=np.float32)
        for i,(f,) in enumerate(batch_buffer):
            batch_frames[i,:f.shape[0]] = f
        return batch_frames, batch_utt_lengths

    batch_ptr = 0
    batch_buffer = batch_size * [None]

    for frames in stream:
        batch_buffer[batch_ptr] = frames
        batch_ptr += 1

        if batch_ptr == batch_size:
            yield construct_batch(batch_buffer)
            batch_ptr = 0

    if batch_ptr > 0:
        batch_buffer = batch_buffer[:batch_ptr]
        yield construct_batch(batch_buffer)

def batched_sort(stream,buffer_size=100):
    batch = buffer_size * [None]
    batch_ptr = 0
    for x in stream:
        batch[batch_ptr] = x
        batch_ptr += 1
        if batch_ptr == buffer_size:
            batch.sort(key=lambda x:x[0].shape[0])
            for x in batch: yield x
            batch_ptr = 0

    batch = batch[:batch_ptr]
    batch.sort(key=lambda x:x[0].shape[0])
    for x in batch: yield x


@config.option("batch_size", "Number of utterances per batch.",type=config.int)
def batched_training_stream(batch_size):
    stream = training_stream()
    sort_buffer = 200
#    stream = batched_sort(stream,buffer_size=sort_buffer)
    batched_stream = batched_utterances(stream,batch_size=batch_size)
    batched_stream = data_io.buffered_random(batched_stream,
                        buffer_items= 2 * max(sort_buffer/batch_size,1))
    return batched_stream

@config.option("training_frame_files","Files for training frames.",
                type=config.file,nargs='+')
def training_stream(training_frame_files):
    split_streams = create_split_streams(training_frame_files)
    split_streams = [ s for s in split_streams ]
    stream = data_io.random_select_stream(*split_streams)
#    stream = data_io.buffered_random(stream,buffer_items=200)
    return stream

@config.option("validation_frame_files","Files for validation frames.",
                type=config.file,nargs='+')
def validation_stream(validation_frame_files):
    from itertools import chain
    split_streams = create_split_streams(validation_frame_files)
    stream = chain(*split_streams)
    stream = batched_utterances(stream,batch_size=1)
    return stream
