#!/bin/bash
gmmdir=exp/tri3
dir=exp_pdnn/dnn_fbank_python

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`

structure="360:1024:1024:1024:1024:$num_pdfs"

python theano-kaldi/pretrain.py \
	--frames-file $dir/pkl/train.pklgz \
	--labels-file $dir/pkl/train_lbl.pklgz \
	--structure $structure \
	--output-file $dir/pretrain.pkl \
	--minibatch 128 --max-epochs 5

python theano-kaldi/train.py \
	--frames-file $dir/pkl/train.pklgz \
	--labels-file $dir/pkl/train_lbl.pklgz \
	--structure $structure \
	--pretrain-file $dir/pretrain.pkl \
	--temporary-file $dir/tmp.dnn.pkl \
	--output-file $dir/dnn.pkl \
	--minibatch 128 --max-epochs 50

theano-kaldi/decode_dnn.sh --nj 1 \
	$gmmdir/graph $dir/data/test \
	${gmmdir}_ali $dir/decode_test
