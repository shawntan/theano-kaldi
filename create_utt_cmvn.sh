#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh
dir=exp/dnn_lda_tk_feedforward
num_jobs=20
data_dir=/home/gautam/Work/aurora-sim/lda/train-lda/data-lda/
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84

[ -d $dir/utt_cmvn_pkl ]  || mkdir -p $dir/utt_cmvn_pkl
feat_transform="copy-feats ark:- ark,t:-"
time $TK_DIR/prepare_pickle.sh $num_jobs \
    $data_dir/dev_0330 \
    $ali_dir \
    $dir/utt_cmvn_pkl/dev\
    $dir/_log/split \
    "$feat_transform" || exit 1;

time $TK_DIR/prepare_pickle.sh $num_jobs \
    $data_dir/train_si84_multi \
    $ali_dir \
    $dir/utt_cmvn_pkl/train \
    $dir/_log/split \
    "$feat_transform" || exit 1;




