#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

data_dir=data-fmllr-tri3
# GMM model for alignments
gmmdir=exp/tri3
ali_dir=${gmmdir}_ali

# Output folder
dir=exp/dnn_fmllr_tk_feedforward

# Create directories
[ -d $dir ]      || mkdir -p $dir
[ -d $dir/data ] || mkdir -p $dir/data
[ -d $dir/pkl ]  || mkdir -p $dir/pkl


# Settings
num_jobs=20
norm_vars=true
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.


# Initial preprocessing for input features
[ -f $dir/feature_transform ] || \
    copy-feats scp:$data_dir/train/feats.scp ark:- \
    | compute-cmvn-stats ark:- - \
    | cmvn-to-nnet --binary=false - $dir/feature_transform  || exit 1;

# Reading -> Transforming -> Writing to pickle
feat_transform="\
nnet-forward $dir/feature_transform ark:- ark:- \
"

[ -f $dir/pkl/train.00.pklgz ] ||\
	time $TK_DIR/prepare_pickle.sh $num_jobs \
    $data_dir/train \
    $ali_dir \
    $dir/pkl/train \
    $dir/_log/split \
    "$feat_transform" || exit 1;

# Training of the nnet.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

input_dim=$(( 11 * $(copy-feats scp:$data_dir/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -) ))

discriminative_structure="$input_dim:1024:1024:1024:1024:1024:1024:$num_pdfs"
echo $discriminative_structure
model_name=nosplice
# Look at using log-normal distribution for the distribution of x

[ -f $dir/pretrain.${model_name}.pkl ] || \
    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/pretrain_sda.py \
        --training-frame-files      ${frame_files[@]:2} \
        --validation-frame-files    ${frame_files[@]:0:2} \
        --structure                 $discriminative_structure \
        --batch-size 128 --max-epochs 5 \
        --output-file $dir/pretrain.${model_name}.pkl


[ -f $dir/discriminative.${model_name}.pkl ] || \
    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train.py \
        --structure $discriminative_structure \
        --training-frame-files ${frame_files[@]:1} \
        --training-label-files ${label_files[@]:1} \
        --validation-frame-files ${frame_files[@]:0:1} \
        --validation-label-files ${label_files[@]:0:1} \
        --max-epochs 50 \
        --batch-size 128 \
        --improvement-threshold 0.999 \
        --weights-file   $dir/pretrain.${model_name}.pkl \
        --learning-file  $dir/discriminative.${model_name}.learning \
        --temporary-file $dir/discriminative.${model_name}.tmp \
        --output-file    $dir/discriminative.${model_name}.pkl \
        --initial-learning-rate  0.1 \
        --momentum 0.9 \
        --log -

for set in dev test
do
    python_posteriors="THEANO_FLAGS=device=gpu0 \
        python $TK_DIR/nnet_forward.py \
        --structure         $discriminative_structure \
        --weights-file      $dir/discriminative.${model_name}.pkl \
        --class-counts-file $dir/decode_${set}_${model_name}/class.counts"

    feats="copy-feats scp:$data_dir/$set/feats.scp ark:- \
        | $feat_transform \
        | $python_posteriors"

    $TK_DIR/decode_dnn.sh \
        --nj 1 \
        --acwt 0.2 \
        $gmmdir/graph $data_dir/${set}\
        ${gmmdir}_ali $dir/decode_${set}_${model_name}\
        "$feats"
done
