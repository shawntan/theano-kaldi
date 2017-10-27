#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# Arguments
num_jobs=$1
data_dir=$2
ali_dir=$(pwd)/$3
output_prefix=$4
log_dir=$5
feat_transform=$6



# Create temporary directory
tmp_dir=$(mktemp -d)
mkdir -p $tmp_dir
mkdir -p $log_dir
[ -f $log_dir/rand ] || dd if=/dev/urandom of=$log_dir/rand count=1024
data_ali_file=$tmp_dir/full.ali
data_feats_file=$tmp_dir/feats.scp
feats_file=$data_dir/feats.scp
spk2utt_file=$data_dir/spk2utt

total_lines=$(wc -l <${spk2utt_file})
((lines_per_file = (total_lines + num_jobs - 1) / num_jobs))
cat $data_dir/spk2utt | cut -d' ' -f1 | shuf --random-source=$log_dir/rand | split -d --lines=${lines_per_file} - "$tmp_dir/spkrs."
 
ls $tmp_dir/spkrs.* | xargs -n 1 -P $num_jobs sh -c '
filename=$1
echo "Starting job... $filename"
idx=${filename##*.}
{
	echo "Starting on split $idx."
	
	cat '"$feats_file"' \
		| grep -F -f $filename \
		| copy-feats scp:- ark:- \
		| '"$feat_transform"' \
		| python2 -u "'$TK_DIR'/pickle_ark_stream.py" "'"$output_prefix"'.$idx.pklgz"

	gunzip -c $( ls '"$ali_dir"'/ali.*.gz | sort -V ) \
		| ali-to-pdf "'"$ali_dir"'/final.mdl" ark:- ark,t:- | sort \
		| grep -F -f $filename \
		| python2 "'$TK_DIR'/pickle_ali.py" "'"$output_prefix"'_lbl.$idx.pklgz"

	gunzip -c $( ls '"$ali_dir"'/ali.*.gz | sort -V ) \
		| ali-to-phones --per-frame "'"$ali_dir"'/final.mdl" ark:- ark,t:- \
		| grep -F -f $filename \
		| python2 "'$TK_DIR'/pickle_ali.py" "'"$output_prefix"'_phn.$idx.pklgz"

} > "'$log_dir'/split.$idx.log" 2>&1
echo "Done."
' fnord
PYTHONPATH=theano-kaldi/ python -c "import sys;import data_io;[ n for n,_,_ in data_io.stream('${output_prefix}.00.pklgz','${output_prefix}_lbl.00.pklgz',with_name=True)]"

rm -rf $tmp_dir
