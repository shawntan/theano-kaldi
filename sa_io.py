import config
import data_io
    
import theano
import theano.tensor as T
import theano_toolkit 
import numpy as np
import math
import sys
import random

def utterance_speaker_mapping(utt2spk_file):
    mapping = dict(lines.strip().split()
                for lines in open(utt2spk_file))
    spkr_id = { spkr:i for i,spkr in enumerate(sorted(set(mapping.values()))) }
    for k in mapping:
        mapping[k] = spkr_id[mapping[k]]
    return mapping

def frame_speaker_stream(stream,utt2spk):
    for tup in stream:
        name = tup[0]
        frames = tup[1]
        speaker = utt2spk[name]
        ids = np.empty(frames.shape[0],dtype=np.int32)
        ids.fill(8)
        yield tup[1:] + (ids,)

def speaker_grouped_stream(frames_files):
    streams = [ data_io.stream(f,with_name=True) for f in frames_files ]
    stream_next = [ s.next() for s in streams ]

    frames_buf = speakers_buf = None
    frame_count = 0
    while len(streams) > 0:
        stream_idx = random.randint(0,len(streams)-1)
        try:
            group = []

            name,frames = stream_next[stream_idx]
            batch_speaker = speaker = name.split('_')[0]
            while speaker == batch_speaker:
                group.append((name,frames))

                stream_next[stream_idx] = streams[stream_idx].next()

                name,frames = stream_next[stream_idx]
                speaker = name.split('_')[0]
        except StopIteration:
            streams = streams[:stream_idx] + streams[stream_idx+1:]
            stream_next = stream_next[:stream_idx] + stream_next[stream_idx+1:]
        if len(group) > 0: yield group

def randomised_speaker_groups(grouped_stream,speaker_ids,
        buffer_size=2**17,
        validation_set=None,
        validation_utt_count=1):
    if validation_set == None:
        validation_set = config.args.validation_frames_file
    import gzip,os
    frames_buf = None
    speakers_buf = None

    frame_count = 0

    if not os.path.isfile(validation_set):
        validation_file = gzip.open(validation_set,'w')
    else:
        validation_file = None
    
    for group in grouped_stream:
        speaker = group[0][0].split('_')[0]
        group_start_frame = frame_count
        if validation_file: pickle.dump(group[0],validation_file,2)
        for name,frames in group[1:]:
            if frames_buf is None:
                frames_buf = np.empty((buffer_size,frames.shape[1]),dtype=np.float32)
                speakers_buf = np.empty((buffer_size,),dtype=np.int32)

            if frame_count + frames.shape[0] > buffer_size:
                if frame_count > group_start_frame:
#                    print "(shuffle %d,%d)"%(group_start_frame,frame_count),
                    np.random.shuffle(frames_buf[group_start_frame:frame_count])
#                print "yield"
                yield frames_buf,speakers_buf,frame_count
                group_start_frame = frame_count = 0

#            print speaker,
#            frames[:,-1] = speaker_ids[speaker]
            frames_buf[frame_count:frame_count+frames.shape[0]] = frames
            speakers_buf[frame_count:frame_count+frames.shape[0]] = speaker_ids[speaker]
            frame_count = frame_count + frames.shape[0]

        if frame_count > group_start_frame:
#            print "(shuffle %d,%d)"%(group_start_frame,frame_count),
            np.random.shuffle(frames_buf[group_start_frame:frame_count])

    
    yield frames_buf,speakers_buf,frame_count
    if validation_file: validation_file.close()
    

def utterance_random_stream(frames_files,labels_files):
    streams = [ data_io.stream(f,l,with_name=True) for f,l in zip(frames_files,labels_files) ]
    while len(streams) > 0:
        stream_idx = random.randint(0,len(streams)-1)
        try:
            yield streams[stream_idx].next()
        except StopIteration:
            streams = streams[:stream_idx] + streams[stream_idx+1:]
