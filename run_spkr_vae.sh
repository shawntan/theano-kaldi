#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments

# Output folder
dir=exp/dnn_lda_tk_feedforward

# Settings
num_jobs=20
norm_vars=true

# Create fbank data set.

data_dir=/home/gautam/Work/aurora-sim/lda/train-lda/data-lda/
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84
# Reading -> Transforming -> Writing to pickle
feat_transform=" nnet-forward $dir/feature_transform ark:- ark,t:- "
# Training of the nnet.
input_dim=$(( 11 * `copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -` ))
num_spkrs=$(cat data/train_si84_multi/spk2utt | wc -l)

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

model_name='acoustic32speaker32'
THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_utterance_speaker.py \
    --frames-files            ${frame_files[@]:2}    \
    --validation-frames-files ${frame_files[@]:0:2}    \
    --pooling-method          max \
    --utt2spk-file            "data/train_si84_multi/utt2spk" \
    --output-file             $dir/semi_sup_speaker.pkl \
    --log - #$dir/_log/train_${model_name}.log

