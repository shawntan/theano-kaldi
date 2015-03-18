#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

dataset=test
canon_dir=exp/dnn_fbank_constraint
lat_dir=$canon_dir/decode_${dataset}_4
working_dir=$canon_dir/sa
gmmdir=exp/tri3

splice_opts=`cat $canon_dir/splice_opts 2>/dev/null` # frame-splicing options.
num_pdfs=`gmm-info $gmmdir/final.mdl | grep pdfs | awk '{print $NF}'`
structure="$(cat $canon_dir/input_dim):1024:1024:1024:1024:1024:1024:$num_pdfs"
echo "Num PDFs $num_pdfs"
mkdir -p $working_dir

#copy-feats scp:$canon_dir/data/$dataset/feats.scp ark:- \
#	| add-deltas --delta-order=$(cat $canon_dir/delta_order) ark:- ark:- \
#	| splice-feats $splice_opts ark:- ark:- \
#	| nnet-forward $canon_dir/feature_transform ark:- ark,t:-  \
#	| python2 $TK_DIR/pickle_ark_stream.py $canon_dir/pkl/${dataset}.pklgz $canon_dir/input_dim
#
lattice-best-path ark:"gunzip -c $lat_dir/lat.*.gz|" ark,t:/dev/null ark,t:- 2> /dev/null \
	| ali-to-pdf $gmmdir/final.mdl ark:- ark,t:- \
	| python2 $TK_DIR/picklise_lbl.py - $canon_dir/pkl/${dataset}_lbl.pklgz

layer=4
mkdir -p $canon_dir/sa/layer_${layer}
python $TK_DIR/train_sa.py \
	--frames-file $canon_dir/pkl/${dataset}.pklgz \
	--labels-file $canon_dir/pkl/${dataset}_lbl.pklgz \
	--structure $structure \
	--pretrain-file $canon_dir/dnn_${layer}.pkl \
	--output-file   $canon_dir/sa/layer_${layer}/ \
	--minibatch 128 --max-epochs 20 \
	--constraint-layer $layer --constraint-weight 0.0125

$TK_DIR/decode_dnn_sa.sh --nj 1 \
	--scoring-opts "--min-lmwt 1 --max-lmwt 8" \
	--norm-vars true \
	$gmmdir/graph $canon_dir/data/test \
	${gmmdir}_ali $canon_dir/decode_test_sa_$layer

#python2 $TK_DIR/picklise_lbl.py $ali_dir $dir/pkl/train_lbl.pklgz
