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
feat_transform="nnet-forward $dir/feature_transform ark:- ark,t:-"

# Training of the nnet.
num_pdfs=`gmm-info $ali_dir/final.mdl | grep pdfs | awk '{print $NF}'`
frame_files=($dir/pkl/train.?*.pklgz)
label_files=($dir/pkl/train_lbl.?*.pklgz)
input_dim=`copy-feats scp:$dir/data/train/feats.scp ark:- | eval $feat_transform | feat-to-dim ark:- -`
discriminative_structure="440:2048:2048:2048:2048:2048:2048:2048:$num_pdfs"

for method in {max,average}
do
    experiment_name=$method
    experiment_dir=$dir/vae/$experiment_name
    mkdir -p $experiment_dir
    [ -f $experiment_dir/generative.pkl ] ||
        THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_utterance_speaker.py \
            --frames-files            ${frame_files[@]:2}    \
            --validation-frames-files ${frame_files[@]:0:2}    \
            --utt2spk-file            "data/train_si84_multi/utt2spk" \
            --pooling-method          $method \
            --output-file             $experiment_dir/generative.pkl \
            --log $experiment_dir/everything.log

    [ -f $dir/augmented_cntk.pkl ] ||
        THEANO_FLAGS=device=gpu0 python -u $TK_DIR/train_speaker_module.py \
            --structure                 $discriminative_structure \
            --speaker-structure         $input_dim:1024:1024:32 \
            --pooling-method            $method \
            --canonical-model           $dir/cntk_model.pkl \
            --frames-files              ${frame_files[@]:2} \
            --labels-files              ${label_files[@]:2} \
            --validation-frames-files   ${frame_files[@]:0:2} \
            --validation-labels-files   ${label_files[@]:0:2} \
            --vae-model                 $experiment_dir/generative.pkl \
            --output                    $experiment_dir/augmented_cntk.pkl \
            --log $experiment_dir/everything.log

    for set in test
    do
        python_posteriors="\
            python $TK_DIR/nnet_forward_speaker_module.py \
            --structure             $discriminative_structure \
            --speaker-structure     $input_dim:1024:1024:32 \
            --pooling-method        $method \
            --model                 $experiment_dir/augmented_cntk.pkl \
            --vae-model             semi_sup.pkl \
            --class-counts          '$dir/decode_test_cntk/class.counts'"

        feat2pos="$feat_transform | $python_posteriors"

        $TK_DIR/decode.sh --nj 8 --acwt 0.1 --config conf/decode_dnn.config \
            /home/gautam/Work/aurora-sim/lda/train-lda/tri3a_dnn/graph_tgpr_5k \
            $dir/data/${set} \
            $experiment_dir/decode_${set} \
            "$feat2pos"\
            $dir
    done
done
