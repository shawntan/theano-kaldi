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
# Initial preprocessing for input features
# Reading -> Transforming -> Writing to pickle
feat_transform="copy-feats ark:- ark,t:-"

# Training of the nnet.
frame_files=($dir/utt_cmvn_pkl/train.?*.pklgz)
label_files=($dir/utt_cmvn_pkl/train_lbl.?*.pklgz)
val_frame_files=($dir/utt_cmvn_pkl/dev.?*.pklgz)
val_label_files=($dir/utt_cmvn_pkl/dev_lbl.?*.pklgz)

for method in max
do
    for batch_size in 10
    do
        for learning_rate in 2e-4
        do
            for gradient_clip in 20
            do
                experiment_name=utterance-mean.layers-3.method-${method}.batch_size-${batch_size}.learning_rate-${learning_rate}.gradient_clip-${gradient_clip}
                experiment_dir=$dir/calibrating_vae/$experiment_name
                echo $experiment_dir
#                epochs=$(( 5 * $batch_size / 10 ))
                mkdir -p $experiment_dir
                THEANO_FLAGS=device=gpu1 python -u $TK_DIR/train_utterance_speaker.py \
                    --frames-files            ${frame_files[@]}    \
                    --utt2spk-file            "data/train_si84_multi/utt2spk" \
                    --validation-frames-files ${val_frame_files[@]}    \
                    --validation-utt2spk-file "data/dev_0330/utt2spk" \
                    --pooling-method          $method \
                    --output-file             $experiment_dir/generative.pkl \
                    --epochs 400 \
                    --learning-curve $experiment_dir/learning_curve \
                    --batch-size $batch_size --learning-rate $learning_rate --gradient-clip $gradient_clip \
                    --log - #$experiment_dir/everything.log
            done
        done
    done
done
