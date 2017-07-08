#!/bin/bash
feat_dim=40
context=5
speaker_latent=100
acoustic_latent=100
activation=softplus
echo $(( $feat_dim + 2 * ( $feat_dim * $context ) ))
python2 analyse_utterances.py \
	--batch-size 500 \
	--training-frame-files pkl/dev.00.pklgz \
	--validation-frame-files pkl/dev.00.pklgz \
    --left-context $context --right-context $context \
    --input-dimension $(( $feat_dim + 2 * ( $feat_dim * $context ) )) \
    --speaker-structure  1024:1024:$speaker_latent \
    --acoustic-structure 1024:1024:$acoustic_latent \
    --decoder-structure  1024 \
    --pooling-method     max      \
    --activation-function $activation \
    --shared-structure 1024 \
    --model-file pkl/generative.pkl
