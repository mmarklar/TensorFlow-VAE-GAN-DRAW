import math
import os
import collections

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("updates_per_epoch", 200, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

FLAGS = flags.FLAGS

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def encoder(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor


def decoder(input_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from 
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:]))
        input_sample = mean + epsilon * stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()).tensor, mean, stddev


def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper

    Args:
        mean: 
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor: tensor produces by decoder 
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

def read_data_set(filename):
    VALIDATION_SIZE=99
    TEST_SIZE = 1000
    EXAMPLE_COUNT = 8523
    SIZE = 784

    data = np.load(filename)
    print data.shape
    # This is hardcoded for now but needs to change
    data = data.reshape(EXAMPLE_COUNT, SIZE)
    validation_classes = data[:VALIDATION_SIZE]
    train_classes = data[VALIDATION_SIZE:-TEST_SIZE]
    test_classes = data[-TEST_SIZE:]
    print validation_classes.shape
    print train_classes.shape
    print test_classes.shape

    validation = mnist.DataSet(validation_classes, validation_classes, np.uint8)
    train = mnist.DataSet(train_classes, validation_classes, np.uint8)
    test = mnist.DataSet(test_classes, validation_classes, np.uint8)
    return Datasets(train=train, validation=validation, test=test)

class VAEModel():
    def __init__(self):
        self.data_directory = os.path.join(FLAGS.working_directory, "MNIST")
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.save_path = FLAGS.working_directory + '/save.ckpt'
        self.mnist = read_data_set("/tmp/vae/converted_java.npy")

        self.input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28])

        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            with pt.defaults_scope(phase=pt.Phase.train):
                with tf.variable_scope("model") as scope:
                    self.output_tensor, self.mean, self.stddev = decoder(encoder(self.input_tensor))

            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model", reuse=True) as scope:
                    self.sampled_tensor, _, _ = decoder()

        self.vae_loss = get_vae_cost(self.mean, self.stddev)
        self.rec_loss = get_reconstruction_cost(self.output_tensor, self.input_tensor)

        self.loss = self.vae_loss + self.rec_loss

        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
        self.train = pt.apply_optimizer(self.optimizer, losses=[self.loss])

        self.init = tf.initialize_all_variables()
        
        self.saver = tf.train.Saver()
    
    def learn(self, sess):
        sess.run(self.init)

        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval = FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                print("Update %s" % i)
                x, _ = self.mnist.train.next_batch(FLAGS.batch_size)
                _, loss_value = sess.run([self.train, self.loss], {self.input_tensor: x})
                training_loss += loss_value

            training_loss = training_loss / \
                (FLAGS.updates_per_epoch * 28 * 28 * FLAGS.batch_size)

            print("Loss %f" % training_loss)
                
        path = self.saver.save(sess, self.save_path)
        print("Model saved to %s" % path)

    def sample(self, sess):
        sess.run(self.init)
        self.saver.restore(sess, self.save_path)
        print "Model restored"

        imgs = sess.run(self.sampled_tensor)
        for k in range(FLAGS.batch_size):
            imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,
                   imgs[k].reshape(28, 28))
