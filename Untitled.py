
# coding: utf-8

# In[1]:

import model
import theano
import theano.tensor as T
params = {}
feedforward = model.build_feedforward(params,input_size=1,layer_sizes=[1]*6,output_size=1)
X = T.matrix('X')
hiddens,outputs = feedforward(X)
prop = theano.function(
    inputs=[X],
    outputs=hiddens[1:]
    )


# In[ ]:

import gzip
import cPickle as pickle
with gzip.open('../train.00.pklgz') as f:
    name,frames = pickle.load(f)


# In[ ]:

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure(figsize=(15,10))

def run(t):
    for i in xrange(6):
        ax = plt.subplot(2,3,i+1)
        activation = hiddens[t].T[i]
        activation_grid = activation.reshape(32,32)
        ax.imshow(activation_grid,cmap=cm.Reds,interpolation="nearest",vmax=1,vmin=0)
        
for x in range(1,7):
    model.load('../dnn.constraint.%d.pkl'%x,params)
    hiddens = np.dstack(prop(frames))
    ani = animation.FuncAnimation(fig, run, frames.shape[0], repeat=False)
    ani.save("animation-%d.mp4"%x, fps=10,bitrate=128)

