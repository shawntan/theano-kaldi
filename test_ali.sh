#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh
ali_dir=/home/gautam/Work/aurora-sim/lda/train-lda/tri2b_multi_ali_si84/
gunzip -c $( ls $ali_dir/ali.*.gz | sort -V ) \
		| ali-to-pdf "$ali_dir/final.mdl" ark:- ark,t:- 


