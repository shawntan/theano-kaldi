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
feat_transform="\
nnet-forward $dir/feature_transform ark:- ark,t:- \
"
# Training of the nnet.
num_pdfs=`gmm-info $ali_dir/final.mdl | grep pdfs | awk '{print $NF}'`

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`

discriminative_structure="96:2048:2048:2048:$num_pdfs"
model_name=latent_features
# Look at using log-normal distribution for the distribution of x

#[ -f $dir/pretrain.${model_name}.pkl ] || \
#    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/pretrain_discriminative.py \
#    --X-files                 ${frame_files[@]:2}    \
#    --Y-files                 ${label_files[@]:2}    \
#    --validation-frames-files ${frame_files[@]:0:2}  \
#    --validation-labels-files ${label_files[@]:0:2}  \
#    --structure               $discriminative_structure \
#    --output-file           $dir/pretrain.${model_name}.pkl \
#    --minibatch 256 \
#    --log - #$dir/_log/train_${model_name}.log


[ -f $dir/discriminative.${model_name}.pkl ] || \
    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_vae_features.py \
    --X-files                 ${frame_files[@]:1}    \
    --Y-files                 ${label_files[@]:1}    \
    --validation-frames-files ${frame_files[@]:0:1}  \
    --validation-labels-files ${label_files[@]:0:1}  \
    --structure               $discriminative_structure \
    --temporary-file          $dir/discriminative.${model_name}.pkl.tmp \
    --output-file             $dir/discriminative.${model_name}.pkl \
    --learning-file           $dir/discriminative.${model_name}.learning\
    --vae-model               $dir/new_fangled.pkl \
    --minibatch 256 --max-epochs 25  \
    --learning-rate 0.08 \
    --learning-rate-decay "0.5" \
    --learning-rate-minimum "1e-6" \
    --improvement-threshold "0.999" \
    --log - #$dir/_log/train_${model_name}.log


for set in test
do
        python_posteriors="\
            python $TK_DIR/nnet_forward_vae_features.py \
            --structure             $discriminative_structure \
            --model                 $dir/discriminative.${model_name}.pkl \
            --vae-model             $dir/new_fangled.pkl \
            --class-counts          '$dir/decode_test_cntk/class.counts'"

        feat2pos="$feat_transform | $python_posteriors"

        $TK_DIR/decode.sh --nj 8 --acwt 0.1 --config conf/decode_dnn.config \
            /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
            $dir/data/${set} \
            $dir/decode_${set}_latent_features \
            "$feat2pos"\
            $dir
done
