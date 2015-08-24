#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments
gmmdir=exp/tri3
ali_dir=${gmmdir}_ali

# Output folder
dir=exp/dnn_fbank_tk_feedforward

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

[ -f $dir/pkl/train.00.pklgz ] ||\
   	time $TK_DIR/prepare_pickle.sh $num_jobs \
	$dir/data/train \
	$ali_dir \
	$dir/pkl/train \
	$dir/_log/split \
	"$feat_transform" || exit 1;

# Training of the nnet.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`
structure="$input_dim:1024:1024:1024:1024:1024:1024:$num_pdfs"
model_name=split

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

[ -f $dir/pretrain.pkl ] || \
	python $TK_DIR/pretrain_sda.py\
	--frames-files ${frame_files[@]:1} \
	--structure $structure \
	--output-file $dir/pretrain.pkl \
	--minibatch 128 --max-epochs 20

[ -f $dir/dnn.${model_name}.pkl ] || \
	python -u $TK_DIR/train.py \
	--frames-files				${frame_files[@]:1} \
	--labels-files				${label_files[@]:1} \
	--validation-frames-file	${frame_files[0]}   \
	--validation-labels-file	${label_files[0]}   \
	--structure					$structure \
	--pretrain-file				$dir/pretrain.pkl \
	--temporary-file			$dir/dnn.${model_name}.pkl.tmp \
	--output-file				$dir/dnn.${model_name}.pkl \
	--minibatch 128 --max-epochs 200

for set in dev test
do
	python_posteriors="THEANO_FLAGS=device=gpu0 \
		python $TK_DIR/nnet_forward.py \
		--structure '$structure' \
		--model '$dir/dnn.${model_name}.pkl' \
		--class-counts '$dir/decode_${set}_${model_name}/class.counts'"

	feats="copy-feats scp:$dir/data/$set/feats.scp ark:- \
		| $feat_transform \
		| $python_posteriors"

	$TK_DIR/decode_dnn.sh --nj 1 \
		--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
		--norm-vars true \
		$gmmdir/graph $dir/data/${set}\
		${gmmdir}_ali $dir/decode_${set}_${model_name}\
		"$feats"
done
