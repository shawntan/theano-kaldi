#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh
dir=exp/dnn_fmllr_tk_feedforward
steps/align_fmllr.sh --nj 20 --cmd "run.pl"  data/dev data/lang exp/tri3 $dir/tri3_ali_dev
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
feat_transform="\
	splice-feats $splice_opts ark:- ark:- |\
	nnet-forward $dir/feature_transform ark:- ark,t:- \
"
ls -al $dir/_fmllr/dev/feats.scp
ls -al $dir/tri3_ali_dev/
$TK_DIR/prepare_pickle.sh 20 \
	$dir/_fmllr/dev/feats.scp \
	$dir/tri3_ali_dev \
	$dir/pkl/dev \
	$dir/_log/split \
	"$feat_transform" || exit 1;

