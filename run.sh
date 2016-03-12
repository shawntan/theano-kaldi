#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments

# Output folder
dir=exp/dnn_lda_tk_feedforward

# Create directories
[ -d $dir ]      || mkdir -p $dir
[ -d $dir/data ] || mkdir -p $dir/data
[ -d $dir/pkl ]  || mkdir -p $dir/pkl


# Settings
num_jobs=20
norm_vars=true

# Create fbank data set.

data_dir=/home/gautam/Work/aurora-sim/lda/train-lda/data-lda/
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84
echo "Copying directories..."
[ -d $dir/data/train ] || \
   cp -r $data_dir/train_si84_multi/ $dir/data/train/
[ -d $dir/data/dev ]   || \
   cp -r $data_dir/dev_0330/         $dir/data/dev/
[ -d "$dir/data/test" ]  || \
    cp -r $data_dir/test_eval92/      $dir/data/test/

# Initial preprocessing for input features
echo "Feature transform.."
[ -f $dir/feature_transform ] || \
    copy-feats scp:$dir/data/train/feats.scp ark:- \
    | compute-cmvn-stats ark:- - \
    | cmvn-to-nnet --binary=false - $dir/feature_transform  || exit 1;

# Reading -> Transforming -> Writing to pickle
feat_transform="\
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
num_pdfs=`gmm-info $ali_dir/final.mdl | grep pdfs | awk '{print $NF}'`

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`

discriminative_structure="440:2048:2048:2048:2048:2048:2048:2048:$num_pdfs"
model_name=nosplice
# Look at using log-normal distribution for the distribution of x

#[ -f $dir/pretrain.${model_name}.pkl ] || \
    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/pretrain_discriminative.py \
    --X-files                 ${frame_files[@]:2}    \
    --Y-files                 ${label_files[@]:2}    \
    --validation-frames-files ${frame_files[@]:0:2}  \
    --validation-labels-files ${label_files[@]:0:2}  \
    --structure               $discriminative_structure \
    --output-file           $dir/pretrain.${model_name}.pkl \
    --minibatch 512 \
    --log - #$dir/_log/train_${model_name}.log


#[ -f $dir/discriminative.${model_name}.pkl ] || \
    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train.py \
    --X-files                 ${frame_files[@]:1}    \
    --Y-files                 ${label_files[@]:1}    \
    --validation-frames-files ${frame_files[@]:0:1}  \
    --validation-labels-files ${label_files[@]:0:1}  \
    --structure               $discriminative_structure \
    --temporary-file          $dir/discriminative.${model_name}.pkl.tmp \
    --output-file             $dir/discriminative.${model_name}.pkl \
    --learning-file           $dir/discriminative.${model_name}.learning\
    --minibatch 256 --max-epochs 200  \
    --learning-rate "0.08" \
    --learning-rate-decay "0.5" \
    --learning-rate-minimum "1e-6" \
    --improvement-threshold "0.99" \
    --pretrain-file           $dir/pretrain.${model_name}.pkl \
    --log - #$dir/_log/train_${model_name}.log

python_posteriors="THEANO_FLAGS=device=gpu1 \
    python $TK_DIR/nnet_forward_cntk.py \
    --structure    $discriminative_structure \
    --model        $dir/cntk_model.pkl \
    --class-counts '$dir/decode_test_cntk/class.counts'"

#( $TK_DIR/decode.sh --nj 8 --acwt 0.1 --config conf/decode_dnn.config \
#    /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
#    $dir/data/test \
#    $dir/decode_test_cntk \
#    "copy-feats ark:- ark,t:- | $python_posteriors" & ) &

for set in test
do
    python_posteriors="THEANO_FLAGS=device=gpu1 \
        python $TK_DIR/nnet_forward.py \
        --structure    $discriminative_structure \
        --model        $dir/discriminative.${model_name}.pkl \
        --class-counts '$dir/decode_test_cntk/class.counts'"

    feat2pos="$feat_transform | $python_posteriors"

    $TK_DIR/decode.sh --nj 8 --acwt 0.1 --config conf/decode_dnn.config \
        /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
        $dir/data/${set} \
        $dir/decode_${set} \
        "$feat2pos"
done
