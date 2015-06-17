#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments
gmmdir=exp/tri3
ali_dir=${gmmdir}_ali

# Output folder
dir=exp/dnn_fmllr_tk_feedforward

# Create directories
[ -d $dir ]        || mkdir -p $dir
[ -d $dir/data ]   || mkdir -p $dir/data
[ -d $dir/pkl ]    || mkdir -p $dir/pkl

# Settings
num_jobs=20
echo "--left-context=5 --right-context=5" > $dir/splice_opts
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.

data_fmllr=$dir/_fmllr
[ -d $data_fmllr ] || (
	mkdir -p $dir/_fmllr 

	set=train
	steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "run.pl" \
		   --transform-dir $ali_dir \
		   $data_fmllr/$set data/$set $gmmdir $dir/log $dir/data || exit 1

	for set in dev test
	do
		steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "run.pl" \
			   --transform-dir $gmmdir/decode_${set} \
			   $data_fmllr/$set data/$set $gmmdir $dir/log $dir/data || exit 1
	done
)

train_data_dir=$data_fmllr/train
# Reading -> Transforming -> Writing to pickle
[ -f $dir/feature_transform ] || \
	copy-feats scp:$train_data_dir/feats.scp ark:- \
	| splice-feats $splice_opts ark:- ark:- \
	| compute-cmvn-stats ark:- - \
	| cmvn-to-nnet --binary=false - $dir/feature_transform  || exit 1;

feat_transform="\
	splice-feats $splice_opts ark:- ark:- |\
	nnet-forward $dir/feature_transform ark:- ark,t:- \
"

[ -f $dir/pkl/train.00.pklgz ] || \
   	time $TK_DIR/prepare_pickle.sh $num_jobs \
	$train_data_dir/feats.scp \
	$ali_dir \
	$dir/pkl/train \
	$dir/_log/split \
	"$feat_transform" || exit 1;

# Training of the nnet.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
input_dim=`copy-feats scp:$train_data_dir/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`
structure="$input_dim:1024:1024:1024:1024:1024:1024:$num_pdfs"
frame_files=($dir/pkl/train.*.pklgz)
label_files=($dir/pkl/train_lbl.*.pklgz)
phoneme_files=($dir/pkl/train_phn.*.pklgz)
#[ -f $dir/pretrain.${model_name}.pkl ] || \
#	python $TK_DIR/pretrain_sda_em.py\
#	--frames-files  ${frame_files[@]:1} \
#	--labels-files  ${label_files[@]:1} \
#	--phoneme-files ${phoneme_files[@]:1} \
#	--gmm-param-dir $dir/pretrain.${model_name}_params/ \
#	--structure $structure \
#	--output-file $dir/pretrain.${model_name}.pkl \
#	--minibatch 128 --max-epochs 20 \
#	--constraint-layer $layer --constraint-coeff $ccoeff 
for layer in -1
do
	for ccoeff in {0.1,0.5}
	do
		for surface in norm
		do
			model_name=phoneme_gaussian_tsne_layer-${layer}_coeff-${ccoeff}-init_${surface}
			mkdir -p $dir/$model_name/
			mkdir -p $dir/$model_name/pretrain_params
			mkdir -p $dir/$model_name/log

			[ -f $dir/class.counts ] ||\
				ali-to-pdf $ali_dir/final.mdl "ark:gunzip -c $ali_dir/ali.*.gz |" ark:- \
				| analyze-counts --binary=false ark:- $dir/class.counts || exit 1;
			for phn in {0..47}	
			do
				for set in dev test
				do

					feats="copy-feats scp:$data_fmllr/$set/feats.scp ark:- \
						| $feat_transform \
						| python2 $TK_DIR/nnet_forward_mute.py $structure $dir/$model_name/dnn.pkl $dir/class.counts $phn"

					[ -d $dir/$model_name/decode_${set}_mute_$phn ] || \
						$TK_DIR/decode_dnn.sh --nj 1 \
						--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
						--norm-vars true \
						$gmmdir/graph $data_fmllr/${set} \
						${gmmdir}_ali $dir/$model_name/decode_${set}_mute_$phn \
						"$feats"
				done
			done
		done
	done
done
