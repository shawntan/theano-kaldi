#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh
gmmdir=exp/mono
ali_dir=${gmmdir}_ali
dir=exp/dnn_fmllr_tk_feedforward
num_jobs=20
data_fmllr=$dir/_fmllr

output_prefix=$dir/pkl/train
rm -rf ${output_prefix}_ctx.*.pklgz
[ -f ${output_prefix}_ctx.00.pklgz ] || {
	feats_file=$data_fmllr/train/feats.scp 
	log_dir=$dir/_log/split
	total_lines=$(wc -l <${feats_file})
	((lines_per_file = (total_lines + num_jobs - 1) / num_jobs))

	tmp_dir=$(mktemp -d)
	mkdir -p $tmp_dir
	gunzip -c $( ls $ali_dir/ali.*.gz | sort -V ) \
		| ali-to-phones --write-lengths $ali_dir/final.mdl ark:- ark,t:- \
		| shuf --random-source=$log_dir/rand \
		| split -d --lines=${lines_per_file} - "$tmp_dir/full.phn."

	ls $tmp_dir/full.phn.* | xargs -n 1 -P $num_jobs sh -c '
	filename=$1
	echo "Starting job... $filename"
	idx=${filename##*.}
	{
		echo "Starting on split $idx."
		cat "'$tmp_dir'/full.phn.$idx" | python2 "'$TK_DIR'/parse_ali_phone_lengths.py" "'"$output_prefix"'_ctx.$idx.pklgz"
	} > "'$log_dir'/split_ctx.$idx.log" 2>&1
	echo "Done."
	' fnord

PYTHONPATH=theano-kaldi-fmllr/ python <<END
import sys
import data_io
for _,a,b in data_io.stream('${output_prefix}_phn.00.pklgz','${output_prefix}_ctx.00.pklgz',with_name=True):
	assert(a.shape[0] == len(b))
END
	rm -rf $tmp_dir
}

