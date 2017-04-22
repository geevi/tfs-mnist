from tfs import *
from mnist import *
import models
import tensorflow as tf

flags.DEFINE_string('mnist_path', '/data4/girish.varma/mnist/mnist.pkl', "MNIST pickle file")
flags.DEFINE_string('arc', 'MLP', "Model class name.")

def main(_):

    FLAGS.project = "mnist"

    mnist = MNIST(FLAGS.mnist_path)

    model = find_class_by_name(FLAGS.arc, [models])(mnist)

    ctrl = init_tf(coord = True, saver = True, writer = True)

    training_loop(ctrl, model, test=True)
    

if __name__ == "__main__":
    tf.app.run()
