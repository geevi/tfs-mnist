from tfs import *

import tensorflow as tf
import cPickle as pkl

class MNIST(object):

    def input_tensor(self, data, name):
        with tf.variable_scope(name):
            images = tf.convert_to_tensor(data[0], dtype=tf.float32)
            labels = tf.convert_to_tensor(data[1], dtype = tf.int32)
            q = tf.train.slice_input_producer([images, labels], shuffle=True, capacity=FLAGS.B)
            images, labels = tf.train.batch([q[0], q[1]],
                        batch_size = FLAGS.B,
                        capacity = 2*FLAGS.B,
                        allow_smaller_final_batch = True,
                        num_threads=2)
            return {'images': images, 'labels': tf.one_hot(labels, 10)}


    def __init__(self, pkl_file):
        data = pkl.load(open(pkl_file))
        self.train =  self.input_tensor(data[0], 'train_input')
        self.val = self.input_tensor(data[1], 'val_input')
