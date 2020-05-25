import inspect
import os

import numpy as np
import tensorflow as tf
import time
from layers.recurrent import hgru_bn_for_shared_gn as hgru


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def __call__(
            self,
            rgb,
            ff_reuse=tf.AUTO_REUSE,
            train=False,
            up_to=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        training = train
        self.ff_reuse = ff_reuse
        self.train = train
        self.input = rgb

        # Convert RGB to BGR
        self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        if up_to == 'conv1_2':
            return self.conv1_2
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        if up_to == 'conv2_2':
            return self.conv2_2
        layer_hgru = hgru.hGRU(
            'fgru_0',
            x_shape=self.conv2_2.get_shape().as_list(),
            timesteps=8,
            h_ext=3,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': ff_reuse, 'constrain': False},
            train=training)
        self.fgru_0 = layer_hgru.build(self.conv2_2)
        self.pool2 = self.max_pool(self.fgru_0, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        if up_to == 'conv3_3':
            return self.conv3_3
        layer_hgru = hgru.hGRU(
            'fgru_1',
            x_shape=self.conv3_3.get_shape().as_list(),
            timesteps=8,
            h_ext=3,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': ff_reuse, 'constrain': False},
            train=training)
        self.fgru_1 = layer_hgru.build(self.conv3_3)
        self.pool3 = self.max_pool(self.fgru_1, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        if up_to == 'conv4_3':
            return self.conv4_3
        layer_hgru = hgru.hGRU(
            'fgru_2',
            x_shape=self.conv4_3.get_shape().as_list(),
            timesteps=8,
            h_ext=1,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': ff_reuse, 'constrain': False},
            train=training)
        self.fgru_2 = layer_hgru.build(self.conv4_3)
        self.pool4 = self.max_pool(self.fgru_2, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        if up_to == 'conv5_3':
            return self.conv5_3
        layer_hgru = hgru.hGRU(
            'fgru_3',
            x_shape=self.conv5_3.get_shape().as_list(),
            timesteps=8,
            h_ext=1, 
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': ff_reuse, 'constrain': False},
            train=training)
        self.fgru_3 = layer_hgru.build(self.conv5_3)
        # self.fgru_3 = layer_hgru.build(self.max_pool(self.conv5_3, name='pool5'))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name=name)

    def conv_layer(
            self,
            bottom,
            name,
            learned=False,
            shape=False,
            apply_relu=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, learned=learned, shape=shape)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, learned=learned, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)

            if apply_relu:
                relu = tf.nn.relu(bias)
            else:
                relu = bias
            return relu

    def get_conv_filter(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    shape=shape,
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.variance_scaling)
            else:
                return tf.get_variable(
                    name='%s_kernel' % name,
                    initializer=self.data_dict[name][0],
                    trainable=self.train)

    def get_bias(self, name, learned=False, shape=None):
        with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
            if learned:
                return tf.get_variable(
                    name='%s_bias' % name,
                    shape=[shape[-1]],
                    dtype=self.dtype,
                    trainable=self.train,
                    initializer=tf.initializers.zeros)
            else:
                return tf.get_variable(
                    name='%s_bias' % name,
                    initializer=self.data_dict[name][1],
                    trainable=self.train)
