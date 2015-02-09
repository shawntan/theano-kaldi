#!/bin/bash
gmmdir=exp/tri3
dir=exp_pdnn/dnn_fbank_connect_all
ali_dir=${gmmdir}_ali
norm_vars=true # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh

splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`

structure="440:1024:1024:1024:1024:1024:1024:$num_pdfs"
echo $structure > structure
mkdir -p $dir
mkdir -p $dir/pkl
set=train

data_dir=$dir/data/$set
time apply-cmvn \
	--norm-vars=$norm_vars \
	--utt2spk=ark:$data_dir/utt2spk \
	scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp \
	ark:- | \
	splice-feats $splice_opts ark:- ark,t:-  | \
	python2 theano-kaldi/pickle_ark_stream.py $dir/pkl/$set.pklgz

python2 theano-kaldi/picklise_lbl.py $ali_dir $set $dir/pkl/${set}_lbl.pklgz

python theano-kaldi/split_dataset.py \
	$dir/pkl/train.pklgz \
	$dir/pkl/train_lbl.pklgz  \
	0.05 \
	$dir/pkl/trn.train.pklgz \
	$dir/pkl/trn.train_lbl.pklgz \
	$dir/pkl/val.train.pklgz \
	$dir/pkl/val.train_lbl.pklgz 

python theano-kaldi/pretrain_sda.py\
	--frames-file $dir/pkl/trn.train.pklgz \
	--labels-file $dir/pkl/trn.train_lbl.pklgz \
	--structure $structure \
	--output-file $dir/pretrain.pkl \
	--minibatch 128 --max-epochs 20


python theano-kaldi/train.py \
	--frames-file $dir/pkl/trn.train.pklgz \
	--labels-file $dir/pkl/trn.train_lbl.pklgz \
	--validation-frames-file $dir/pkl/val.train.pklgz \
	--validation-labels-file $dir/pkl/val.train_lbl.pklgz \
	--structure $structure \
	--pretrain-file $dir/pretrain.pkl \
	--temporary-file $dir/tmp.dnn.pkl \
	--output-file $dir/dnn.pkl \
	--minibatch 256 --max-epochs 100

theano-kaldi/decode_dnn.sh --nj 1 \
	--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
	--norm-vars true \
	$gmmdir/graph $dir/data/test \
	${gmmdir}_ali $dir/decode_test
