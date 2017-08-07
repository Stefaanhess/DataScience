from couzin import couzin_next_step
from helperfunctions import *

import tensorflow as tf
import numpy as np

predict_all = True
folder = 'Graphs/20_80_200/'
meta_graph = folder + 'meta_graph_1'
model = folder + 'my-model_1'

sess = tf.Session()
saver = tf.train.import_meta_graph(meta_graph)
saver.restore(sess, model)


def run_network(aj, agents, norm, history=2):
    if (len(aj.history)>history):
        # TODO: Check this!!!!!!
        track_continuous = np.expand_dims(np.array(aj.history[-history:]), axis=0)
        track_discrete = np.expand_dims(digitize_track(np.array(aj.history[-history:])), axis=0)
        result = sess.run(['generator/gen_output:0'], feed_dict={'track_continuous:0': track_discrete,
                                                                 'track_discrete:0': track_discrete})
        discretization_bins = np.load("Bins.npy")
        bin_array_x = get_bin_means(discretization_bins[0])
        bin_array_y = get_bin_means(discretization_bins[1])        
        x_dir = np.argmax(result[0][0][-1][0])
        y_dir = np.argmax(result[0][0][-1][1])
        predicted_velo = np.array([bin_array_x[x_dir],bin_array_y[y_dir]])
        # update_color(ai, angle_error(ai.velo_temp,predicted_velo))
        if (predict_all == True and len(aj.history)>history):
            aj.velo_temp = predicted_velo
    else:
        couzin_next_step(aj, agents, norm)


def digitize_track(track):
    discretization_bins = np.load("Bins.npy")
    for feature_i in range(74):
        track[:, feature_i] = np.digitize(track[:, feature_i], discretization_bins[feature_i], right=True)
    return track.astype(np.int32)

