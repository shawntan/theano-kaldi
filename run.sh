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
    | compute-cmvn-stats ark:- - \
    | cmvn-to-nnet --binary=false - $dir/feature_transform  || exit 1;

# Reading -> Transforming -> Writing to pickle
feat_transform="\
add-deltas --delta-order=$(cat $dir/delta_order) ark:- ark:- |\
nnet-forward $dir/feature_transform ark:- ark:- \
"

[ -f $dir/pkl/train.00.pklgz ] ||\
	time $TK_DIR/prepare_pickle.sh $num_jobs \
    $dir/data/train \
    $ali_dir \
    $dir/pkl/train \
    $dir/_log/split \
    "$feat_transform" || exit 1;

[ -f $dir/pkl/raw_train.00.pklgz ] ||\
	time $TK_DIR/prepare_pickle.sh $num_jobs \
    $dir/data/train \
    $ali_dir \
    $dir/pkl/raw_train \
    $dir/_log/split_raw \
    "copy-feats ark:- ark,t:-" || exit 1;

# Training of the nnet.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`

discriminative_structure="1353:1024:1024:1024:1024:1024:1024:$num_pdfs"
model_name=nosplice.bn
# Look at using log-normal distribution for the distribution of x

#[ -f $dir/pretrain.${model_name}.pkl ] || \
#    THEANO_FLAGS=device=cuda python -u $TK_DIR/pretrain_sda.py \
#        --training-frame-files      ${frame_files[@]:2} \
#        --validation-frame-files    ${frame_files[@]:0:2} \
#        --structure                 $discriminative_structure \
#        --batch-size 128 --max-epochs 5 \
#        --output-file $dir/pretrain.${model_name}.pkl


#[ -f $dir/discriminative.${model_name}.pkl ] || \
    THEANO_FLAGS=device=cuda python -u $TK_DIR/train.py \
        --structure $discriminative_structure \
        --training-frame-files ${frame_files[@]:1} \
        --training-label-files ${label_files[@]:1} \
        --validation-frame-files ${frame_files[@]:0:1} \
        --validation-label-files ${label_files[@]:0:1} \
        --max-epochs 50 \
        --batch-size 1024 \
        --improvement-threshold 0.999 \
        --learning-file  $dir/discriminative.${model_name}.learning \
        --temporary-file $dir/discriminative.${model_name}.tmp \
        --output-file    $dir/discriminative.${model_name}.pkl \
        --initial-learning-rate 0.2 \
        --momentum 0.9 \
        --log -

THEANO_FLAGS=device=cuda python2 -u theano-kaldi/compute_statistics.py \
    --structure $discriminative_structure \
    --training-frame-files ${frame_files[@]} \
    --training-label-files ${label_files[@]} \
    --model-file        $dir/discriminative.${model_name}.pkl \
    --augmented-file    $dir/discriminative.${model_name}.with_stats.pkl \
#  --weights-file   $dir/pretrain.nosplice.pkl \
for set in dev test
do
    python_posteriors="THEANO_FLAGS=device=cuda,optimizer=None \
        python $TK_DIR/nnet_forward.py \
        --structure         $discriminative_structure \
        --model-file      $dir/discriminative.${model_name}.with_stats.pkl \
        --class-counts-file $dir/decode_${set}_${model_name}/class.counts"

    feats="copy-feats scp:$dir/data/$set/feats.scp ark:- \
        | $feat_transform \
        | $python_posteriors"

    $TK_DIR/decode_dnn.sh \
        --nj 1 \
        --acwt 0.2 \
        $gmmdir/graph $dir/data/${set}\
        ${gmmdir}_ali $dir/decode_${set}_${model_name}\
        "$feats"
done
