#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments
gmmdir=exp/tri3
ali_dir=${gmmdir}_ali

# Output folder
dir=exp/dnn_fbank_tk_feedforward_vae

# Create directories
[ -d $dir ]      || mkdir -p $dir
[ -d $dir/data ] || mkdir -p $dir/data
[ -d $dir/pkl ]  || mkdir -p $dir/pkl


# Settings
num_jobs=20
norm_vars=true
echo "--left-context=5 --right-context=5" > $dir/splice_opts
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
echo 2 > $dir/delta_order
{
echo "--use-energy=true"
echo "--num-mel-bins=40"
} > conf/fbank.conf

# Create fbank data set.
[ -d $dir/_fbank ] || (
for set in train dev test
do
	cp -r data/$set $dir/data/$set
	rm -rf $dir/data/$set/{cmvn,feats}.scp $dir/data/$set/split*
	steps/make_fbank.sh --fbank-config conf/fbank.conf --cmd "run.pl" --nj $num_jobs $dir/data/$set $dir/_log $dir/_fbank || exit 1;
done
)

# Initial preprocessing for input features
[ -f $dir/feature_transform ] || \
	copy-feats scp:$dir/data/train/feats.scp ark:- \
	| add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- \
	| splice-feats $splice_opts ark:- ark:- \
	| compute-cmvn-stats ark:- - \
	| cmvn-to-nnet --binary=false - $dir/feature_transform  || exit 1;

# Reading -> Transforming -> Writing to pickle
feat_transform="\
add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- |\
splice-feats $splice_opts ark:- ark:- |\
nnet-forward $dir/feature_transform ark:- ark,t:- \
"
[ -f $dir/pkl/train.00.pklgz ] || \
	time $TK_DIR/prepare_pickle.sh $num_jobs \
	$dir/data/train \
	$ali_dir \
	$dir/pkl/train \
	$dir/_log/split_train \
	"$feat_transform" || exit 1;

[ -f $dir/pkl/test.00.pklgz ] || \
	time $TK_DIR/prepare_pickle.sh $num_jobs \
	$dir/data/test \
	$ali_dir \
	$dir/pkl/test \
	$dir/_log/split_test \
	"$feat_transform" || exit 1;

[ -f $dir/pkl/dev.00.pklgz ] || \
	time $TK_DIR/prepare_pickle.sh $num_jobs \
	$dir/data/dev \
	$ali_dir \
	$dir/pkl/dev \
	$dir/_log/split_dev \
	"$feat_transform" || exit 1;


# Training of the nnet.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`
gen_structure="$input_dim:1024:512"
dis_structure="512:1024:1024:1024:1024:$num_pdfs"
model_name=split

frame_files=($dir/pkl/train.*.pklgz)
label_files=($dir/pkl/train_lbl.*.pklgz)

#python -u $TK_DIR/test_vae.py \
#	--frames-files ${frame_files[@]:1:1} \
#	--labels-files ${label_files[@]:1:1} \
#	--structure "$input_dim:1024:1024:512" \
#	--output-file $dir/pretrain.pkl \
#	--minibatch 128 --max-epochs 5

#[ -f $dir/generative_sa.pkl ] || \
	python -u $TK_DIR/train_sa_vae.py \
	--frames-files ${frame_files[@]} \
	--generative-structure $gen_structure \
	--validation-frames-file $dir/pkl/gen_val.pklgz   \
	--output-file  $dir/generative_sa.pkl \
	--spk2utt-file $dir/data/train/spk2utt \
	--minibatch 256 --max-epochs 20

#[ -f $dir/generative_sa_train.pkl ] || \
#	python -u $TK_DIR/adapt_sa_vae.py \
#	--frames-files			$dir/pkl/train.*.pklgz \
#	--generative-structure	$gen_structure \
#	--validation-frames-file $dir/pkl/gen_val_train.pklgz   \
#	--generative-model $dir/generative_sa.pkl \
#	--output-file  $dir/generative_sa_train.pkl \
#	--spk2utt-file $dir/data/train/spk2utt \
#	--minibatch 256 --max-epochs 20


#[ -f $dir/discriminative_sa.pkl ] || \
	python -u $TK_DIR/train_sa.py \
	--frames-files				${frame_files[@]:1} \
	--labels-files				${label_files[@]:1} \
	--validation-frames-file	${frame_files[0]}   \
	--validation-labels-file	${label_files[0]}   \
	--generative-model			$dir/generative_sa.pkl \
	--generative-structure		$gen_structure \
	--discriminative-structure	$dis_structure \
	--temporary-file $dir/tmp.discriminative_sa.pkl \
	--output-file    $dir/discriminative_sa.pkl \
	--spk2utt-file $dir/data/train/spk2utt \
	--minibatch 128 --max-epochs 200


#[ -f $dir/generative_sa_dev.pkl ] || \
	python -u $TK_DIR/adapt_sa_vae.py \
	--frames-files			$dir/pkl/dev.*.pklgz \
	--generative-structure	$gen_structure \
	--validation-frames-file $dir/pkl/gen_val_dev.pklgz   \
	--generative-model $dir/generative_sa.pkl \
	--output-file  $dir/generative_sa_dev.pkl \
	--spk2utt-file $dir/data/dev/spk2utt \
	--minibatch 256 --max-epochs 20


for set in dev
do
	for sample in 1
	do
		python_posteriors="THEANO_FLAGS=device=gpu0 \
			python $TK_DIR/nnet_forward_sa.py \
			--generative-structure		'$gen_structure' \
			--discriminative-structure	'$dis_structure' \
			--generative-model		'$dir/generative_sa_${set}.pkl' \
			--discriminative-model	'$dir/discriminative_sa.pkl' \
			--spk2utt-file 			'$dir/data/$set/spk2utt' \
			--class-counts			'$dir/decode_${set}_${model_name}/class.counts'"

		feats="copy-feats scp:$dir/data/$set/feats.scp ark:- \
			| $feat_transform \
			| $python_posteriors"

		$TK_DIR/decode_dnn.sh --nj 1 \
			--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
			--norm-vars true \
			$gmmdir/graph $dir/data/${set}\
			${gmmdir}_ali $dir/decode_${set}_sa\
			"$feats"
	done
done
