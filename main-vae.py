'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from vaemodel import VAEModel

if __name__ == "__main__":
    with tf.Session() as sess:
        vae = VAEModel()
        vae.learn(sess)
