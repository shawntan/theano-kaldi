import data_io
import math,sys,gzip
import cPickle as pickle
if __name__ == "__main__":
	frame_file = sys.argv[1]
	label_file = sys.argv[2]
	percent_split = float(sys.argv[3])
	frame_file_1 = sys.argv[4]
	label_file_1 = sys.argv[5]
	frame_file_2 = sys.argv[6]
	label_file_2 = sys.argv[7]
	
	sample = int(math.ceil(1/percent_split))
	with gzip.open(frame_file_1,'wb') as feat_f_1,\
		 gzip.open(frame_file_2,'wb') as feat_f_2,\
		 gzip.open(label_file_1,'wb') as lbls_f_1,\
		 gzip.open(label_file_2,'wb') as lbls_f_2:
		count = 1
		validation_count = 0
		training_count = 0
		for name,feats,lbls in data_io.stream(frame_file,label_file,with_name=True):
			if count % sample == 0:
				print "Dumped %d instances, writing to validation..."%count
				pickle.dump((name,feats),feat_f_2,protocol=2)
				pickle.dump((name,lbls),lbls_f_2,protocol=2)
				validation_count += 1
			else:
				pickle.dump((name,feats),feat_f_1,protocol=2)
				pickle.dump((name,lbls),lbls_f_1,protocol=2)
				training_count += 1
			count += 1

	print "Dumped %d to validation and %d to training"%(validation_count,training_count)
