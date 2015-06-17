#!/bin/bash
TK_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
[ -f $TK_DIR/path.sh ] && . $TK_DIR/path.sh

# Arguments
num_jobs=$1
feats_file=$2
ali_dir=$3
output_prefix=$4
log_dir=$5
feat_transform=$6

total_lines=$(wc -l <${feats_file})
((lines_per_file = (total_lines + num_jobs - 1) / num_jobs))

# Create temporary directory
tmp_dir=$(mktemp -d)
mkdir -p $tmp_dir
mkdir -p $log_dir
[ -f $log_dir/rand ] || dd if=/dev/urandom of=$log_dir/rand count=1024
data_ali_file=$tmp_dir/full.ali
data_feats_file=$tmp_dir/feats.scp

# Create files.
echo "Creating alignments file."
gunzip -c $( ls $ali_dir/ali.*.gz | sort -V ) \
	| ali-to-pdf $ali_dir/final.mdl ark:- ark,t:- \
	> $data_ali_file
wc -l $data_ali_file
echo "Filtering features file."
cat $feats_file \
	| grep -F -f <(cut -f1 -d' ' $data_ali_file ) \
	> $data_feats_file
wc -l $data_feats_file
echo "Splitting and shuffling files."
shuf --random-source=$log_dir/rand $data_ali_file   | split -d --lines=${lines_per_file} - "$tmp_dir/full.ali."
shuf --random-source=$log_dir/rand $data_feats_file | split -d --lines=${lines_per_file} - "$tmp_dir/feats.scp."

ls $tmp_dir/feats.scp.* | xargs -n 1 -P $num_jobs sh -c '
filename=$1
echo "Starting job... $filename"
idx=${filename##*.}
{
	echo "Starting on split $idx."
	copy-feats scp:"$filename" ark:- \
		| '"$feat_transform"' \
		| python2 -u "'$TK_DIR'/pickle_ark_stream.py" "'"$output_prefix"'.$idx.pklgz"
	cat "'$tmp_dir'/full.ali.$idx" | python2 "'$TK_DIR'/pickle_ali.py" "'"$output_prefix"'_lbl.$idx.pklgz"
} > "'$log_dir'/split.$idx.log" 2>&1
echo "Done."
' fnord
#rm -rf $tmp_dir
PYTHONPATH=theano-kaldi/ python -c "import sys;import data_io;[ n for n,_,_ in data_io.stream('${output_prefix}.00.pklgz','${output_prefix}_lbl.00.pklgz',with_name=True)]"
