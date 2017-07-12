#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

dir=exp/dnn_fbank_tk_feedforward
gmmdir=exp/tri3
frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
feat_transform="\
add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- |\
nnet-forward $dir/feature_transform ark:- ark:- \
"
input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`

model_name=nosplice.bn

discriminative_structure="1353:1024:1024:1024:1024:1024:1024:$num_pdfs"
THEANO_FLAGS=device=cpu python2 -u theano-kaldi/compute_statistics.py \
    --structure $discriminative_structure \
    --training-frame-files ${frame_files[@]} \
    --training-label-files ${label_files[@]} \
    --model-file        $dir/discriminative.${model_name}.pkl \
    --augmented-file    $dir/discriminative.${model_name}.with_stats.pkl \
