#!/usr/bin/env python
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.recurrent import hgru_bn_for_shared_v1 as hgru


def build_model(data_tensor, reuse, training, output_shape):
    """Create the hgru from Learning long-range..."""
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    with tf.variable_scope('cnn', reuse=reuse):
        # Add input
        in_emb = conv.input_layer(
            X=data_tensor,
            reuse=reuse,
            training=training,
            features=20,
            conv_activation=[tf.nn.relu, None],
            conv_kernel_size=7,
            pool_kernel_size=[1, 2, 2, 1],
            pool_kernel_strides=[1, 2, 2, 1],
            pool=True,
            name='l0')
        layer_hgru = hgru.hGRU(
            'hgru_1',
            x_shape=in_emb.get_shape().as_list(),
            timesteps=8,
            h_ext=20,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux={'reuse': False, 'constrain': False},
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
            output_shape=output_shape)
    extra_activities = {
        'activity': h2
    }
    return activity, extra_activities

