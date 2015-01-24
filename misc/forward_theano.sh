dir=exp_pdnn/dnn_fbank_noisy/
data_dir=$dir/data/test
norm_vars=false  # when doing cmvn, whether to normalize variance; has to be consistent with build_nnet_pfile.sh

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
#ls $dir/nnet.pkl
time apply-cmvn \
	--norm-vars=$norm_vars \
	--utt2spk=ark:$data_dir/utt2spk \
	scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp \
	ark:- | \
   	splice-feats $splice_opts ark:- ark,t:-  | \
 	THEANO_FLAGS=device=gpu0 python2 nnet_forward.py $dir/nnet.pkl $dir/decode_test/class.counts > nnet_forward.py.out
time apply-cmvn \
	--norm-vars=$norm_vars \
	--utt2spk=ark:$data_dir/utt2spk \
	scp:$data_dir/cmvn.scp scp:$data_dir/feats.scp \
	ark:- | \
   	splice-feats $splice_opts ark:- ark:-  | \
  	nnet-forward --class-frame-counts=$dir/decode_test/class.counts --apply-log=true --no-softmax=false $dir/dnn.nnet ark:- ark,t:-   > nnet_forward.out

wc -l nnet_forward.py.out
wc -l nnet_forward.out

grep '\[$' nnet_forward.py.out | wc -l
grep '\[$' nnet_forward.out | wc -l

grep '\]$' nnet_forward.py.out | wc -l
grep '\]$' nnet_forward.out | wc -l
