#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh
#set -e
echo $PATH


gmmdir=exp/tri3
orig_dir=exp/dnn_fbank_tk_feedforward
dir=exp/dnn_fbank_connect_all
ali_dir=${gmmdir}_ali
#norm_vars=true # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh
#
#echo "--left-context=5 --right-context=5" > $dir/splice_opts
#splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
#echo 2 > $dir/delta_order
#
#
#mkdir -p $dir
#echo \
#"--use-energy=true
#--num-mel-bins=40" > conf/fbank.conf
#for set in train dev test
#do
#	cp -r data/$set $dir/data/$set
#	rm -rf $dir/data/$set/{cmvn,feats}.scp $dir/data/$set/split*
#	steps/make_fbank.sh --fbank-config conf/fbank.conf --cmd "run.pl" --nj 10 $dir/data/$set $dir/_log $dir/_fbank || exit 1;
##	steps/compute_cmvn_stats.sh                $dir/data/$set $dir/_log $dir/_fbank || exit 1;
#done
#copy-feats scp:$dir/data/train/feats.scp ark:- \
#	| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| compute-cmvn-stats ark:- - \
#	| cmvn-to-nnet --binary=false - $dir/feature_transform 


#mkdir -p $dir/pkl
#time copy-feats scp:$dir/data/train/feats.scp ark:- \
#	| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| nnet-forward $dir/feature_transform ark:- ark,t:-  \
#	| python2 $TK_DIR/pickle_ark_stream.py $dir/pkl/train.pklgz $dir/input_dim

#python2 $TK_DIR/picklise_lbl.py $ali_dir $dir/pkl/train_lbl.pklgz
ln -s ../../$orig_dir/delta_order $dir/delta_order
ln -s ../../$orig_dir/splice_opts $dir/splice_opts
ln -s ../../$orig_dir/data        $dir/data
ln -s ../../$orig_dir/pkl         $dir/pkl
ln -s ../../$orig_dir/input_dim   $dir/input_dim
ln -s ../../$orig_dir/feature_transform $dir/feature_transform

num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
structure="$(cat $dir/input_dim):1024:1024:1024:1024:1024:1024:$num_pdfs"
echo $structure > $dir/structure
#structure=$(cat $dir/structure)

#python $TK_DIR/split_dataset.py \
#	$dir/pkl/train.pklgz \
#	$dir/pkl/train_lbl.pklgz  \
#	0.05 \
#	$dir/pkl/trn.train.pklgz \
#	$dir/pkl/trn.train_lbl.pklgz \
#	$dir/pkl/val.train.pklgz \
#	$dir/pkl/val.train_lbl.pklgz 

#python $TK_DIR/pretrain_sda.py\
#	--frames-file $dir/pkl/trn.train.pklgz \
#	--labels-file $dir/pkl/trn.train_lbl.pklgz \
#	--structure $structure \
#	--output-file $dir/pretrain.pkl \
#	--minibatch 128 --max-epochs 20


#python $TK_DIR/train.py \
#	--frames-file $dir/pkl/trn.train.pklgz \
#	--labels-file $dir/pkl/trn.train_lbl.pklgz \
#	--validation-frames-file $dir/pkl/val.train.pklgz \
#	--validation-labels-file $dir/pkl/val.train_lbl.pklgz \
#	--structure $structure \
#	--temporary-file $dir/tmp.dnn.pkl \
#	--output-file $dir/dnn.pkl \
#	--minibatch 256 --max-epochs 100

python $TK_DIR/train_fine.py \
	--frames-file $dir/pkl/trn.train.pklgz \
	--labels-file $dir/pkl/trn.train_lbl.pklgz \
	--validation-frames-file $dir/pkl/val.train.pklgz \
	--validation-labels-file $dir/pkl/val.train_lbl.pklgz \
	--pretrain-file $orig_dir/pretrain.pkl \
	--structure $structure \
	--temporary-file $dir/tmp.dnn.dynamic_gates.pkl \
	--output-file $dir/dnn.dynamic_gates.pkl \
	--minibatch 128 --max-epochs 200

rm $dir/dnn.pkl
ln -s dnn.adjust_gates.pkl $dir/dnn.pkl
#for output_layer in {0..5}
#do
#	$TK_DIR/decode_dnn.sh --nj 1 \
#		--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
#		--norm-vars true \
#		$gmmdir/graph $dir/data/test \
#		${gmmdir}_ali $dir/decode_test_$output_layer $output_layer
#done
$TK_DIR/decode_dnn.sh --nj 1 \
	--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
	--norm-vars true \
	$gmmdir/graph $dir/data/test \
	${gmmdir}_ali $dir/decode_test "-1" 

