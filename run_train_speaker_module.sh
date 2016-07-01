#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# GMM model for alignments

# Output folder
dir=exp/vae_lda
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84

# Settings
num_jobs=20
norm_vars=true

[ -d $dir/pkl ]  || mkdir -p $dir/pkl

[ -f $dir/pkl/dev.00.pklgz ] || (\
    for dataset in dev_0330 test_eval92 train_si84_multi 
    do
        prefix=$(echo $dataset | cut -d'_' -f1)
        $TK_DIR/prepare_pickle.sh $num_jobs \
            data-mfcc-utt-cmvn/$dataset \
            $ali_dir \
            $dir/pkl/$prefix \
            $dir/_log/${prefix}_split \
            "copy-feats ark:- ark:-" || exit 1;
    done
)

# Training of the nnet.
frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
val_frame_files=($dir/pkl/dev.?*.pklgz)
val_label_files=($dir/pkl/dev_lbl.?*.pklgz)
feat_dim=40
context=5


for method in max
do
    for batch_size in 10
    do
        for learning_rate in 3e-4
        do
            for gradient_clip in 100
            do
                latent_size=64
                experiment_name=frame-estimator.latent-${latent_size}.newsample.method-${method}.batch_size-${batch_size}.learning_rate-${learning_rate}.gradient_clip-${gradient_clip}.nanfilter
                experiment_dir=$dir/calibrating_vae/$experiment_name
                echo $experiment_dir
                mkdir -p $experiment_dir
                device=0
                THEANO_FLAGS=device=gpu${device} python -u $TK_DIR/train_utterance_vae.py \
                    --training-frame-files          ${frame_files[@]:2}    \
                    --validation-frame-files        ${frame_files[@]:0:2}    \
                    --batch-size $batch_size \
                    --left-context $context --right-context $context \
                    --input-dimension $(( $feat_dim + 2 * ( $feat_dim * $context ) )) \
                    --speaker-structure  1024:1024:100 \
                    --pooling-method     max      \
                    --acoustic-structure 1024:1024:$latent_size \
                    --decoder-structure  1024:1024    \
                    --max-epochs 300 \
                    --learning-rate $learning_rate \
                    --gradient-clip $gradient_clip \
                    --temporary-model-file $experiment_dir/generative.pkl.tmp \
                    --temporary-training-file $experiment_dir/generative.pkl.trn \
                    --iteration-log $experiment_dir/learning_curve \
                    --log - #$experiment_dir/everything.log
#                sleep 5
#                THEANO_FLAGS=device=gpu${device} python -u $TK_DIR/train_utterance_vae.py \
#                    --training-frame-files          ${frame_files[@]:2}    \
#                    --validation-frame-files        ${frame_files[@]:0:2}    \
#                    --batch-size 20 \
#                    --left-context $context --right-context $context \
#                    --input-dimension $(( $feat_dim + 2 * ( $feat_dim * $context ) )) \
#                    --speaker-structure  1024:1024:100 \
#                    --pooling-method     max      \
#                    --acoustic-structure 1024:1024:$latent_size \
#                    --decoder-structure  1024:1024    \
#                    --max-epochs 50 \
#                    --learning-rate 1e-4 \
#                    --gradient-clip $gradient_clip \
#                    --temporary-model-file $experiment_dir/generative.pkl \
#                    --temporary-training-file $experiment_dir/generative.pkl.trn.fin \
#                    --previous-model-file $experiment_dir/generative.pkl.tmp \
#                    --previous-training-file $experiment_dir/generative.pkl.trn \
#                    --iteration-log $experiment_dir/learning_curve \
#                    --log - #$experiment_dir/everything.log

            done
        done
    done
done
