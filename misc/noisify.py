from scipy.io import wavfile
import numpy as np
import sys
audio_file = sys.argv[1]
noise_file = sys.argv[2]
snr_db = float(sys.argv[3])
print "Loading",audio_file,"...",
sample_rate,audio_data = wavfile.read(audio_file)
sample_rate,noise_data = wavfile.read(noise_file)

# Take only stuff from first half.
start_time_limit = noise_data.shape[0]/2 - audio_data.shape[0]
start_time = np.random.randint(low=0,high=start_time_limit)
noise_data = noise_data[start_time:start_time+audio_data.shape[0]]


audio_var = np.var(audio_data)
noise_var = np.var(noise_data)
f = audio_var/( (10 **(snr_db/10)) * noise_var)

noise_data = (np.sqrt(f) * noise_data).astype(np.int16)

wavfile.write('%s.noise.wav'%audio_file,sample_rate,audio_data + noise_data)
print "Done."
