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
num_spkrs=$(cat data/train_si84_multi/spk2utt | wc -l)
discriminative_structure="$input_dim:2048:2048:2048:$num_spkrs"

frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)

model_name='acoustic2speaker2'
THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_spkr_vec.py \
    --X-files                 ${frame_files[@]:1}    \
    --Y-files                 ${label_files[@]:1}    \
    --validation-frames-files ${frame_files[@]:0:1}  \
    --utt2spk-file            "data/train_si84_multi/utt2spk" \
    --speaker-structure       $input_dim:1024:256:64:16:2 \
    --acoustic-structure      $input_dim:1024:256:64:16:2 \
    --decoder-structure       16:64:256:1024:$input_dim \
    --temporary-file          $dir/vae.${model_name}.pkl.tmp \
    --output-file             $dir/vae.${model_name}.pkl \
    --learning-file           $dir/vae.${model_name}.learning\
    --minibatch 512 --max-epochs 200  \
    --learning-rate "1e-4" \
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

