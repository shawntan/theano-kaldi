#!/bin/bash
gmmdir=exp/tri3
dir=exp_pdnn/dnn_fbank_constraint
ali_dir=${gmmdir}_ali
norm_vars=true # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh

delta_order=2
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`


structure="1080:1024:1024:1024:1024:1024:1024:$num_pdfs"
echo $structure > structure
mkdir -p $dir
mkdir -p $dir/pkl
ln -s ../dnn_fbank.orig/data $dir/data

set=train
data_dir=$dir/data/$set
feature_transform=$dir/feature_transform


#copy-feats scp:$data_dir/feats.scp ark:- \
#	| add-deltas --delta-order=$delta_order ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| compute-cmvn-stats ark:- - \
#	| cmvn-to-nnet --binary=false - $feature_transform 

#copy-feats scp:$data_dir/feats.scp ark:- \
#	| add-deltas --delta-order=$delta_order ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| nnet-forward $feature_transform ark:- ark,t:-  \
#	| python2 theano-kaldi-2/pickle_ark_stream.py $dir/pkl/$set.pklgz

#python2 theano-kaldi-2/picklise_lbl.py $ali_dir "$dir/pkl/${set}_lbl.pklgz"
##
#python2 theano-kaldi-2/split_dataset.py \
#	$dir/pkl/train.pklgz \
#	$dir/pkl/train_lbl.pklgz  \
#	0.05 \
#	$dir/pkl/trn.train.pklgz \
#	$dir/pkl/trn.train_lbl.pklgz \
#	$dir/pkl/val.train.pklgz \
#	$dir/pkl/val.train_lbl.pklgz 


#python theano-kaldi-2/pretrain_sda.py\
#	--frames-file $dir/pkl/trn.train.pklgz \
#	--labels-file $dir/pkl/trn.train_lbl.pklgz \
#	--structure $structure \
#	--output-file $dir/pretrain.pkl \
#	--minibatch 128 --max-epochs 20


python theano-kaldi-2/train.py \
	--frames-file $dir/pkl/trn.train.pklgz \
	--labels-file $dir/pkl/trn.train_lbl.pklgz \
	--validation-frames-file $dir/pkl/val.train.pklgz \
	--validation-labels-file $dir/pkl/val.train_lbl.pklgz \
	--structure $structure \
	--pretrain-file $dir/pretrain.pkl \
	--temporary-file $dir/tmp.dnn.pkl \
	--output-file $dir/dnn.pkl \
	--minibatch 128 --max-epochs 100

theano-kaldi-2/decode_dnn.sh --nj 1 \
	--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
	--norm-vars true \
	$gmmdir/graph $dir/data/test \
	${gmmdir}_ali $dir/decode_test
