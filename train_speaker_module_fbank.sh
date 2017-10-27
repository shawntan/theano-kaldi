#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments

# Output folder
dir=exp/vae_fbank

# Settings
num_jobs=20
norm_vars=true

[ -d $dir/pkl ]  || ln -s ../dnn_fbank_tk_feedforward/pkl $dir/pkl
# Training of the nnet.
frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
context=0

datadir=data-fbank/train_si84_multi
#feat_dim=$(copy-feats ark:/u/tanjings/kaldi/egs/aurora4/s5_/fbank/raw_fbank_train_si84_multi.1.ark ark:- | feat-to-dim ark:- -)
feat_dim=120
echo input_dim $input_dim
for method in max
do
    for batch_size in 10
    do
        for learning_rate in 5e-4
        do
            for gradient_clip in 1e20
            do
                echo $context $feat_dim $(( $feat_dim + 2 * ( $feat_dim * $context ) ))
                activation=softplus
                speaker_latent=100
                acoustic_latent=100
                experiment_name=fbank.mila
                experiment_dir=$dir/calibrating_vae/$experiment_name
                model_params="--left-context $context --right-context $context \
                    --input-dimension $(( $feat_dim + 2 * ( $feat_dim * $context ) )) \
                    --speaker-structure  512:$speaker_latent \
                    --acoustic-structure 512:$acoustic_latent \
                    --decoder-structure  1024 \
                    --pooling-method     max       \
                    --activation-function $activation \
                    --shared-structure 1024"

                echo $experiment_dir
                mkdir -p $experiment_dir
                export THEANO_FLAGS=device=cuda
                python -u $TK_DIR/train_utterance_vae_shared.py \
                    --training-frame-files   ${frame_files[@]:1}    \
                    --validation-frame-files ${frame_files[@]:0:1}    \
                    --batch-size $batch_size \
                    --max-epochs 900 \
                    $model_params \
                    --learning-rate $learning_rate \
                    --gradient-clip $gradient_clip \
                    --temporary-model-file $experiment_dir/generative.pkl.tmp \
                    --temporary-training-file $experiment_dir/generative.pkl.trn \
                    --iteration-log $experiment_dir/learning_curve \
                    --log - \
                    --iteration-log iteration.log
                sleep 2

                python -u $TK_DIR/train_utterance_vae_shared.py \
                    --training-frame-files   ${frame_files[@]:1}    \
                    --validation-frame-files ${frame_files[@]:0:1}    \
                    $model_params \
                    --batch-size 20 \
                    --max-epochs 100 \
                    --learning-rate 1e-4 \
                    --gradient-clip $gradient_clip \
                    --previous-model-file $experiment_dir/generative.pkl.tmp \
                    --previous-training-file $experiment_dir/generative.pkl.trn \
                    --temporary-model-file $experiment_dir/generative.pkl \
                    --temporary-training-file $experiment_dir/generative.pkl.trn.final \
                    --iteration-log $experiment_dir/learning_curve \
                    --log - #$experiment_dir/everything.log
            done
        done
    done
done
