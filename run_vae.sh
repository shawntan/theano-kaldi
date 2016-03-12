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
# Reading -> Transforming -> Writing to pickle
feat_transform=" nnet-forward $dir/feature_transform ark:- ark,t:- "
# Training of the nnet.
input_dim=$(( 11 * `copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -` ))
num_pdfs=`gmm-info $ali_dir/final.mdl | grep pdfs | awk '{print $NF}'`
discriminative_structure="$input_dim:2048:2048:2048:2048:2048:2048:2048:$num_pdfs"

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

model_name=vae_sep_2d
THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_vae.py \
    --X-files                 ${frame_files[@]:1}    \
    --Y-files                 ${label_files[@]:1}    \
    --validation-frames-files ${frame_files[@]:0:1}  \
    --validation-labels-files ${label_files[@]:0:1}  \
    --structure               $discriminative_structure \
    --temporary-file          $dir/discriminative.${model_name}.pkl.tmp \
    --output-file             $dir/discriminative.${model_name}.pkl \
    --learning-file           $dir/discriminative.${model_name}.learning\
    --minibatch 512 --max-epochs 200  \
    --learning-rate "1e-3" \
    --learning-rate-decay "1" \
    --learning-rate-minimum "1e-10" \
    --improvement-threshold "2" \
    --log - #$dir/_log/train_${model_name}.log
    #--pretrain-file           $dir/pretrain.${model_name}.pkl \
#    --pretrain-file           $dir/discriminative.vae_separate.pkl.tmp \

for set in test
do
    python_posteriors="THEANO_FLAGS=device=gpu0 \
        python $TK_DIR/nnet_forward_vdnn.py \
        --structure    $discriminative_structure \
        --model        $dir/discriminative.${model_name}.pkl \
        --class-counts '$dir/decode_test_cntk/class.counts'"

    feat2pos="$feat_transform | $python_posteriors"
    $TK_DIR/decode.sh --nj 8 --acwt 0.1 --config conf/decode_dnn.config \
        /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
        $dir/data/${set} \
        $dir/decode_${model_name}_${set} \
        "$feat2pos"
done

