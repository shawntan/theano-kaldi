#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments

# Output folder
dir=exp/vae_fbank
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84

# Settings
num_jobs=20
norm_vars=true

[ -d $dir/pkl ]  || mkdir -p $dir/pkl

# Training of the nnet.
frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
val_frame_files=($dir/pkl/dev.?*.pklgz)
val_label_files=($dir/pkl/dev_lbl.?*.pklgz)
feat_dim=40
feat_dim=40
context=9
device=$1
for method in max
do
    for batch_size in 10
    do
        for learning_rate in 5e-4
        do
            for gradient_clip in 1e20
            do
                activation=softplus
                speaker_latent=100
                acoustic_latent=100
                experiment_name=nonsim.adamfix.${activation}.speaker-${speaker_latent}.acoustic-${acoustic_latent}.method-${method}.batch_size-${batch_size}.learning_rate-${learning_rate}.gradient_clip-${gradient_clip}.nanfilter
                experiment_dir=$dir/calibrating_vae/$experiment_name
                model_params="--left-context $context --right-context $context \
                    --input-dimension $(( $feat_dim + 2 * ( $feat_dim * $context ) )) \
                    --speaker-structure  1024:1024:$speaker_latent \
                    --acoustic-structure 1024:1024:$acoustic_latent \
                    --decoder-structure  2048 \
                    --pooling-method     max       \
                    --activation-function $activation \
                    --shared-structure 1024"

                echo $experiment_dir
                mkdir -p $experiment_dir
                export THEANO_FLAGS=device=gpu${device},lib.cnmem=1
                python -u $TK_DIR/train_utterance_vae_shared.py \
                    --training-frame-files          ${frame_files[@]}    \
                    --validation-frame-files        ${val_frame_files[@]}    \
                    --batch-size $batch_size \
                    --max-epochs 900 \
                    $model_params \
                    --learning-rate $learning_rate \
                    --gradient-clip $gradient_clip \
                    --temporary-model-file $experiment_dir/generative.pkl.tmp \
                    --temporary-training-file $experiment_dir/generative.pkl.trn \
                    --iteration-log $experiment_dir/learning_curve \
                    --log - #$experiment_dir/everything.log \
                sleep 2

                python -u $TK_DIR/train_utterance_vae_shared.py \
                    --training-frame-files          ${frame_files[@]}    \
                    --validation-frame-files        ${val_frame_files[@]}    \
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
