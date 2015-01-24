ali_dir=exp/tri3_ali
dir=$1
set=train
data_dir=$dir/data/$set
norm_vars=false  # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh
splice_opts=`cat $dir/splice_opts 2>/dev/null` # frame-splicing options.
#ls $dir/nnet.pkl
mkdir -p $dir/pkl
time apply-cmvn \
	--norm-vars=$norm_vars \
	--utt2spk=ark:$data_dir/utt2spk \
	scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp \
	ark:- | \
   	splice-feats $splice_opts ark:- ark,t:-  | \
 	python2 theano-kaldi/pickle_ark_stream.py $dir/pkl/$set.pklgz
python2 theano-kaldi/picklise_lbl.py $ali_dir $dir/pkl/${set}_lbl.pklgz
