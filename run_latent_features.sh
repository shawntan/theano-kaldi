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
input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`

discriminative_structure="96:2048:2048:2048:2048:2048:2048:$num_pdfs"
model_name=latent_features
# Look at using log-normal distribution for the distribution of x
[ -f $dir/pkl/dev.00.pklgz ] ||\
	time $TK_DIR/prepare_pickle.sh $num_jobs \
    $dir/data/dev \
    /home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_dev_0330/ \
    $dir/pkl/dev \
    $dir/_log/split \
    "$feat_transform" || exit 1;

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
val_frame_files=($dir/pkl/dev.?*.pklgz)
val_label_files=($dir/pkl/dev_lbl.?*.pklgz)

THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_vae_features.py \
    --X-files                 ${frame_files[@]}    \
    --Y-files                 ${label_files[@]}    \
    --validation-frames-files ${val_frame_files[@]}  \
    --validation-labels-files ${val_label_files[@]}  \
    --structure               96:2048:$num_pdfs \
    --temporary-file          $dir/pretrain.layers_1.pkl.tmp \
    --learning-file           $dir/pretrain.layers_1.pkl.learning \
    --vae-model               $dir/new_fangled.pkl \
    --output-file             $dir/pretrain.layers_1.pkl \
    --minibatch 256 --max-epochs 15 \
    --learning-rate 0.1 \
    --learning-rate-decay "0.5" \
    --learning-rate-minimum "1e-6" \
    --improvement-threshold "0.999" \
    --momentum 0 \
    --log - #$dir/_log/train_${model_name}.log


for layers in {2..4}
do
    prev_layers=$(($layers-1))
    hiddens=`printf "2048:%.0s" $(seq $layers)`
    structure="96:${hiddens}${num_pdfs}"
    echo $structure
    echo $dir/pretrain.layers_${prev_layers}.pkl
    echo $dir/pretrain.layers_${layers}.pkl

    THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_vae_features.py \
        --X-files                 ${frame_files[@]}    \
        --Y-files                 ${label_files[@]}    \
        --validation-frames-files ${val_frame_files[@]}  \
        --validation-labels-files ${val_label_files[@]}  \
        --structure               $structure \
        --temporary-file          $dir/pretrain.layers_${prev_layers}.pkl.tmp \
        --learning-file           $dir/pretrain.layers_${prev_layers}.pkl.learning \
        --vae-model               $dir/new_fangled.pkl \
        --pretrain-file           $dir/pretrain.layers_${prev_layers}.pkl \
        --output-file             $dir/pretrain.layers_${layers}.pkl \
        --minibatch 256 --max-epochs 15  \
        --learning-rate 1 \
        --learning-rate-decay "0.5" \
        --learning-rate-minimum "1e-6" \
        --improvement-threshold "0.999" \
        --log - #$dir/_log/train_${model_name}.log
done
[ -f $dir/discriminative.${model_name}.pkl ] || \
THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_vae_features.py \
    --X-files                 ${frame_files[@]}    \
    --Y-files                 ${label_files[@]}    \
    --validation-frames-files ${val_frame_files[@]}  \
    --validation-labels-files ${val_label_files[@]}  \
    --structure               $discriminative_structure \
    --temporary-file          $dir/discriminative.${model_name}.pkl.tmp \
    --learning-file           $dir/discriminative.${model_name}.pkl.learning \
    --vae-model               $dir/new_fangled.pkl \
    --pretrain-file           $dir/pretrain.layers_6.pkl \
    --output-file             $dir/discrminative.${model_name}.pkl \
    --minibatch 256 --max-epochs 15  \
    --learning-rate 1 \
    --learning-rate-decay "0.5" \
    --learning-rate-minimum "1e-6" \
    --improvement-threshold "0.99" \
    --log - #$dir/_log/train_${model_name}.log


for set in test
do
    python_posteriors="THEANO_FLAGS=gpu0 \
        python $TK_DIR/nnet_forward_vae_features.py \
        --structure             $discriminative_structure \
        --model                 $dir/discriminative.${model_name}.pkl.tmp \
        --vae-model             $dir/new_fangled.pkl \
        --class-counts          '$dir/decode_test_cntk/class.counts'"

    feat2pos="$feat_transform | $python_posteriors"

    time $TK_DIR/decode.sh --nj 8 --acwt 0.0833 --config conf/decode_dnn.config \
        /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
        $dir/data/${set} \
        $dir/decode_${set}_latent_features \
        "$feat2pos"\
        $dir
    
done
for x in exp/dnn*/decode_test*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done | send_notification.sh
