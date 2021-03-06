import pickle
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import datetime
from Settings import Settings

### Variables of the algorithm

settings = Settings()
min_track_length = settings.min_track_length
num_discretization_bins = settings.num_discretization_bins
file = settings.outfile
restore = False

num_batches = 5000
num_hidden = 100  # hochsetzen --> mächtigeres Modell
batch_size = 20
folder = settings.folder

def splitChunks(t):
    """
    Small trick to increase the dataset size: A single track may have >500 detections, even after /15 reduction.
    Split every track into multiple tracks of 150 length.
    """
    trackChunks = []
    while (len(t)>=min_track_length+3):
        currentTrack = t[0:min_track_length]
        t = t[min_track_length:len(t)]
        trackChunks = trackChunks + [currentTrack]
    return trackChunks

def split_all_tracks(np_array):
    """
    Splits all single tracks into multiple tracks via the splitChunks method
    """
    list_tracks = np_array.tolist()
    tracks = []
    for track in list_tracks:
        chunks = splitChunks(track)
        tracks = tracks + [chunks]
    return tracks

tracks = np.load(file)
#tracks_1 = np.split(np.array(np_tracks),5,axis=1)
#tracks = np.concatenate(np.array(tracks_1))

# tracks = split_all_tracks(np_tracks) # optional, only if we want to split our tracks

### End of Variables of the algorithm 

print(np.array(tracks).shape)

# optional try to avoid tracks that are too short
tracks = list(filter(lambda t: len(t) > min_track_length + 1, tracks))
tracks = list(map(np.array, tracks))
num_features = tracks[0].shape[-1]

# not used at the moment
def get_discretization_bins(data, bins):
    """
    Bens Function for dynamic bins of same size
    """
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    return cutoffs

# our own static bins function
# TODO min & max substitute with norm
def get_equal_discretization_bins(data, bins):
    """
    create equally sized bins bewteen minium and maximum value,
    excluding start- and endpoint

    Parameters
    ----------
    data : ndarray
        feature data of all agents concatenated
    bins : int
        number of bins
    """
    min_data = min(data)
    max_data = max(data)
    ret_bins = np.linspace(min_data, max_data, bins, endpoint=False)
    ret_bins = np.delete(ret_bins, 0)
    return ret_bins

def discretize_sight(bins):
    ret_bins = np.linspace(0., 1., bins, endpoint=False)
    ret_bins = np.delete(ret_bins, 0)
    return ret_bins

# Find out the bins

discretization_bins = []
concatenated_tracks = np.concatenate(tracks)
for feature_i in range(2):
    discretization_bins.append(get_equal_discretization_bins(concatenated_tracks[:, feature_i], num_discretization_bins))
for feature_i in range(2, num_features):
    discretization_bins.append(discretize_sight(num_discretization_bins))
del(concatenated_tracks)
np.save("Bins",discretization_bins)

# Digitize all values into the bins
def digitize_track(track):
    for feature_i in range(num_features):
        track[:, feature_i] = np.digitize(track[:, feature_i], discretization_bins[feature_i], right=True)
    return track.astype(np.int32)

### MODEL SET UP BEGINS

np.random.shuffle(tracks)
train_tracks = tracks[:int(.8 * len(tracks))]
val_tracks = tracks[int(.8 * len(tracks)):]
print(len(train_tracks), len(val_tracks))

tf.reset_default_graph()

track_continuous = tf.placeholder(tf.float32, shape=(None, min_track_length, num_features), name='track_continuous')
track_discrete = tf.placeholder(tf.int32, shape=(None, min_track_length, num_features), name='track_discrete')

num_outputs = 2

inputs = track_continuous[:, :-1, :]
targets = track_discrete[:, 1:, :num_outputs]

with tf.variable_scope('discriminator'):
    disc_hidden_0, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        inputs, dtype=tf.float32, scope='disc_hidden_0')
    disc_hidden_1, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        disc_hidden_0, dtype=tf.float32, scope='disc_hidden_1')
    disc_hidden_2, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        disc_hidden_1, dtype=tf.float32, scope='disc_hidden_2')
    # add optional classifier here
    
with tf.variable_scope('generator'):
    gen_hidden_2, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        disc_hidden_2, dtype=tf.float32, scope='gen_hidden_2')
    gen_hidden_1_input = tf.concat((gen_hidden_2, disc_hidden_1), axis=2)
    gen_hidden_1, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        gen_hidden_1_input, dtype=tf.float32, scope='gen_hidden_1')
    gen_hidden_0_input = tf.concat((gen_hidden_1, disc_hidden_0), axis=2)
    gen_hidden_0, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden), 
        gen_hidden_0_input, dtype=tf.float32, scope='gen_hidden_0')
    gen_output, _ = tf.nn.dynamic_rnn(
        tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(num_hidden), num_outputs * num_discretization_bins), 
        gen_hidden_0, dtype=tf.float32, scope='gen_features')
    gen_output = tf.reshape(gen_output,
        (tf.shape(gen_output)[0], tf.shape(gen_output)[1], num_outputs, num_discretization_bins), name='gen_output')

# LOSS FUNCTION    
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=gen_output))

optimizer = tf.train.AdamOptimizer(0.005)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    update = optimizer.minimize(loss)

def subsample(track):
    start_idx = np.random.randint(0, track.shape[0] - min_track_length)
    end_idx = start_idx + min_track_length
    return track[start_idx:end_idx]

def data_generator(tracks, size=batch_size):
    while True:
        indices = np.random.choice(list(range(len(tracks))), replace=False, size=size)
        # not shure about this???
        #samples = [tracks[idx] for idx in indices]
        samples = [track for idx, track in enumerate(tracks) if idx in indices]
        sampled_tracks = np.array(list(map(subsample, samples)))
        digitized_tracks = np.array(list(map(lambda t: digitize_track(t), np.copy(sampled_tracks))))
        #print(sampled_tracks)
        #print(digitized_tracks)
        yield sampled_tracks, digitized_tracks

train_gen = data_generator(train_tracks)
val_gen = data_generator(val_tracks)

session = tf.Session()
saver = tf.train.Saver()
session.run(tf.global_variables_initializer())

# xxx
if restore:
    saver.restore(session, folder + 'my-model_1')

train_losses = []
val_losses = []
start_time = time.time()
for batch_idx in range(num_batches):
    samples_continuous, samples_discrete = next(train_gen)
    batch_loss, _ = session.run([loss, update], feed_dict={track_continuous: samples_continuous,
                                              track_discrete: samples_discrete})
    train_losses.append(batch_loss)
    
    samples_continuous, samples_discrete = next(val_gen)
    batch_loss = session.run(loss, feed_dict={track_continuous: samples_continuous,
                                              track_discrete: samples_discrete})
    val_losses.append(batch_loss)
    iter_time = (time.time() - start_time)/(batch_idx+1)
    sys.stdout.write('\r{}: train-logloss: {:.2f}, val-logloss: {:.2f}, etimated time remaining: {}'.format(
        batch_idx, np.mean(train_losses[-100:]), np.mean(val_losses[-100:]), datetime.timedelta(seconds=iter_time*(num_batches-batch_idx))))

saver.save(session, folder + 'my-model_1')
tf.train.export_meta_graph(folder + 'meta_graph_1')

plt.plot(pd.Series(train_losses).rolling(100).mean(), label='train-logloss')
plt.plot(pd.Series(val_losses).rolling(100).mean(), label='val-logloss')
plt.legend()
plt.show()
