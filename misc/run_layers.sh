#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

gmmdir=exp/tri3
#orig_dir=exp/dnn_fbank_tk_feedforward
dir=exp/dnn_fbank_tk_feedforward
ali_dir=${gmmdir}_ali
norm_vars=true # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh

echo "--left-context=5 --right-context=5" > $dir/splice_opts
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
echo 2 > $dir/delta_order


mkdir -p $dir
mkdir -p $dir/data
#ln -s ../../$orig_dir/delta_order $dir/delta_order
#ln -s ../../$orig_dir/splice_opts $dir/splice_opts
#ln -s ../../$orig_dir/data        $dir/data
#ln -s ../../$orig_dir/pkl         $dir/pkl
#ln -s ../../$orig_dir/input_dim   $dir/input_dim
#ln -s ../../$orig_dir/feature_transform $dir/feature_transform

echo \
"--use-energy=true
--num-mel-bins=40" > conf/fbank.conf
#for set in train dev test
#do
#	echo "cp -rv data/$set $dir/data/$set"
#
#	cp -rv data/$set $dir/data/$set
#	rm -rf $dir/data/$set/{cmvn,feats}.scp $dir/data/$set/split*
#	steps/make_fbank.sh --fbank-config conf/fbank.conf --cmd "run.pl" --nj 10 $dir/data/$set $dir/_log $dir/_fbank || exit 1;
##	steps/compute_cmvn_stats.sh                $dir/data/$set $dir/_log $dir/_fbank || exit 1;
#done
#
#copy-feats scp:$dir/data/train/feats.scp ark:- \
#	| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| compute-cmvn-stats ark:- - \
#	| cmvn-to-nnet --binary=false - $dir/feature_transform 
#
#mkdir -p $dir/pkl
#time copy-feats scp:$dir/data/train/feats.scp ark:- \
#	| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| nnet-forward $dir/feature_transform ark:- ark,t:-  \
#	| python2 $TK_DIR/pickle_ark_stream.py $dir/pkl/train.pklgz $dir/input_dim
#
#python2 $TK_DIR/picklise_lbl.py $ali_dir $dir/pkl/train_lbl.pklgz
#

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
echo $structure > $dir/structure
structure=$(cat $dir/structure)

#python $TK_DIR/split_dataset.py \
#	$orig_dir/pkl/train.pklgz \
#	$orig_dir/pkl/train_lbl.pklgz  \
#	0.05 \
#	$orig_dir/pkl/trn.train.pklgz \
#	$orig_dir/pkl/trn.train_lbl.pklgz \
#	$orig_dir/pkl/val.train.pklgz \
#	$orig_dir/pkl/val.train_lbl.pklgz 

#python $TK_DIR/pretrain_sda.py\
#	--frames-file $dir/pkl/trn.train.pklgz \
#	--labels-file $dir/pkl/trn.train_lbl.pklgz \
#	--structure $structure \
#	--output-file $dir/pretrain.pkl \
#	--minibatch 128 --max-epochs 20
hiddens=1024:1024:1024:1024:1024
mv $dir/dnn.pkl $dir/dnn.6.pkl
for layers in 5
do
	structure="$(cat $dir/input_dim):$hiddens:$num_pdfs"
	echo $structure
#	python $TK_DIR/train.py \
#		--frames-file $dir/pkl/trn.train.pklgz \
#		--labels-file $dir/pkl/trn.train_lbl.pklgz \
#		--validation-frames-file $dir/pkl/val.train.pklgz \
#		--validation-labels-file $dir/pkl/val.train_lbl.pklgz \
#		--structure $structure \
#		--pretrain-file $dir/pretrain.pkl \
#		--temporary-file $dir/tmp.dnn.${layers}.pkl \
#		--output-file $dir/dnn.${layers}.pkl \
#		--minibatch 128 --max-epochs 100


	feats="copy-feats scp:$dir/data/test/feats.scp ark:- \
		| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
		| splice-feats $splice_opts ark:- ark:- \
		| nnet-forward $dir/feature_transform ark:- ark,t:- \
		| python2 theano-kaldi/nnet_forward.py $structure $dir/dnn.${layers}.pkl $dir/decode_test_$layers/class.counts"

	echo $feats
	$TK_DIR/decode_dnn.sh --nj 1 \
		--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
		--norm-vars true \
		$gmmdir/graph $dir/data/test \
		${gmmdir}_ali $dir/decode_test_$layers \
		"$feats"

	hiddens=$hiddens:1024
done
