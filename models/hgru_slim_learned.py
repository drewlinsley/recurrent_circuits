#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_for as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    reduction_filters = 24
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = tf.layers.conv2d(
            inputs=data_tensor,
            filters=16,
            activation=tf.nn.elu,
            kernel_size=(7, 7))
        in_emb = tf.layers.max_pooling2d(
            inputs=in_emb,
            pool_size=(4, 4),
            strides=(2, 2))
        in_emb = tf.layers.conv2d(
            inputs=in_emb,
            filters=reduction_filters,
            kernel_size=(7, 7))
        in_emb = in_emb ** 2
        layer_hgru = hgru.hGRU(
            'hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=11,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': False, 'constrain': False, 'nonnegative': True, 'while_loop': False},
            train=training)
        h2 = layer_hgru.build(in_emb)
        h2 = normalization.batch(
            bottom=h2,
            renorm=False,
            name='hgru_bn',
            training=training)
        activity = conv.readout_layer(
            activity=h2,
            reuse=reuse,
            training=training,
            pool_type='max',  # 'select',
            output_shape=output_shape,
            features=reduction_filters)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities
