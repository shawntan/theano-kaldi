Theano-Kaldi
============

A solution for training and decoding neural networks with unique structure
using the Kaldi toolkit.

## Usage
```
cd <dataset>/s5
git clone --recursive git@github.com:shawntan/theano-kaldi.git
theano-kaldi/run.sh
```

## Dataset results

Available dataset results. These will be updated as they are done.

####`timit`
```
%WER 18.6 | 400 15057 | 84.1 11.6 4.3 2.7 18.6 99.8 | -0.938 | fbank/dev
%WER 20.6 | 192 7215 | 82.2 12.7 5.1 2.8 20.6 100.0 | -0.966 | fbank/test
%WER 17.5 | 400 15057 | 84.9 10.7 4.4 2.4 17.5 99.5 | -0.818 | fmllr/dev
%WER 18.4 | 192 7215 | 84.0 11.3 4.7 2.4 18.4 100.0 | -0.801 | fmllr/test
```
