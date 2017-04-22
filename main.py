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

class MLP(BaseModel):

    def __init__(self, dataset):

        net = [
            ['dense', {
                'units' 	: 1000,
                'act'	: 'relu'
            }],
            ['dense', {
                'units'	: 10,
                'act'	: None
            }]
        ]

        self.logits_train = sequential(dataset.train['images'], net, name ='mlp')
        self.logits_val = sequential(dataset.val['images'], net, name = 'mlp', reuse = True)
        args = {
            'y'             : dataset.train['labels'],
            'y_pred'        : self.logits_train,
            'y_val'         : dataset.val['labels'],
            'y_pred_val'    : self.logits_val,
            'rate'          : FLAGS.rate
        }
        self.optimizer, self.train_summary_op, self.val_summary_op, self.global_step = classify(**args)
        self.train_feed = self.val_feed = None





flags.DEFINE_string('mnist_path', '/data4/girish.varma/mnist/mnist.pkl', "Imagenet train folder")




def main(_):

    FLAGS.project = "mnist"

    mnist = MNIST(FLAGS.mnist_path)

    model = MLP(mnist)

    ctrl = init_tf(coord = True, saver = True, writer = True)

    training_loop(ctrl, model, test=True)
    






if __name__ == "__main__":
    tf.app.run()


