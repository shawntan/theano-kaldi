#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

data_dir=data-lda

latent_dir=data-utt_cmvn_vae

for set in train_si84_multi dev_0330 test_eval92
do
    mkdir -p $latent_dir/$set
    cp -v $data_dir/$set/spk2gender $latent_dir/$set/
    cp -v $data_dir/$set/spk2utt    $latent_dir/$set/
    cp -v $data_dir/$set/utt2spk    $latent_dir/$set/
    cp -v $data_dir/$set/text       $latent_dir/$set/
    copy-feats scp:$data_dir/$set/feats.scp ark:-\
        | THEANO_FLAGS=device=gpu0 python2 $TK_DIR/generate_latent_ark.py \
            --vae-model exp/dnn_lda_tk_feedforward/vae/max/generative.pkl \
        | copy-feats ark:- ark,scp:$latent_dir/$set/feats.ark,$latent_dir/$set/feats.scp
    sleep 5
done
