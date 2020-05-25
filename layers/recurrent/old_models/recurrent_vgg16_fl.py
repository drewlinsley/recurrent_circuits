import inspect
import os
import numpy as np
import tensorflow as tf
import time
from collections import OrderedDict
from ops import tf_fun
from layers.recurrent.gn_params import CreateGNParams
from layers.recurrent.gn_params import defaults
from layers.recurrent.gammanet_refactored import GN
from layers.recurrent.gn_recurrent_ops import GNRnOps
from layers.recurrent.gn_feedforward_ops import GNFFOps


class Vgg16(GN, CreateGNParams, GNRnOps, GNFFOps):
    def __init__(
            self,
            vgg16_npy_path,
            train,
            timesteps,
            reuse,
            fgru_normalization_type,
            ff_normalization_type,
            layer_name='recurrent_vgg16',
            ff_nl=tf.nn.relu,
            horizontal_kernel_initializer=tf.initializers.orthogonal(),
            kernel_initializer=tf.initializers.orthogonal(),
            gate_initializer=tf.initializers.orthogonal(),
            train_ff_gate=None,
            train_fgru_gate=None,
            train_norm_moments=None,
            train_norm_params=None,
            train_fgru_kernels=None,
            train_fgru_params=None,
            up_kernel=None,
            stop_loop=False,
            recurrent_ff=False,
            strides=[1, 1, 1, 1],
            pool_strides=[16, 16],  # Because fgrus are every other down-layer
            pool_kernel=[8, 8],
            data_format='NHWC',
            horizontal_padding='SAME',
            ff_padding='SAME',
            aux=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path
        self.data_format = data_format
        self.pool_strides = pool_strides
        self.strides = strides
        self.pool_kernel = pool_kernel
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.horizontal_padding = horizontal_padding
        self.ff_padding = ff_padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        self.horizontal_kernel_initializer = horizontal_kernel_initializer
        self.kernel_initializer = kernel_initializer
        self.gate_initializer = gate_initializer
        self.fgru_normalization_type = fgru_normalization_type
        self.ff_normalization_type = ff_normalization_type
        self.recurrent_ff = recurrent_ff
        self.stop_loop = stop_loop
        self.ff_nl = ff_nl
        self.fgru_connectivity = ''
        self.timesteps = timesteps
        if train_ff_gate is None:
            self.train_ff_gate = self.train
        else:
            self.train_ff_gate = train_ff_gate
        if train_fgru_gate is None:
            self.train_fgru_gate = self.train
        else:
            self.train_fgru_gate = train_fgru_gate
        if train_norm_moments is None:
            self.train_norm_moments = self.train
        else:
            self.train_norm_moments = train_norm_moments
        if train_norm_moments is None:
            self.train_norm_params = self.train
        else:
            self.train_norm_params = train_norm_params
        if train_fgru_kernels is None:
            self.train_fgru_kernels = self.train
        else:
            self.train_fgru_kernels = train_fgru_kernels
        if train_fgru_kernels is None:
            self.train_fgru_params = self.train
        else:
            self.train_fgru_params = train_fgru_params

        default_vars = defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)
        # Store variables in the order they were created. Hack for python 2.x.
        self.variable_list = OrderedDict()
        self.hidden_dict = OrderedDict()

        # Kernel info
        if data_format is 'NHWC':
            self.prepared_pool_kernel = [1] + self.pool_kernel + [1]
            self.prepared_pool_stride = [1] + self.pool_strides + [1]
            self.up_strides = [1] + self.pool_strides + [1]
        else:
            self.prepared_pool_kernel = [1, 1] + self.pool_kernel
            self.prepared_pool_stride = [1, 1] + self.pool_strides
            self.up_strides = [1, 1] + self.pool_strides
        self.sanity_check()
        if self.symmetric_weights:
            self.symmetric_weights = self.symmetric_weights.split('_')

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = tf_fun.interpret_nl(self.recurrent_nl)

        # Set initializers for greek letters
        if self.force_alpha_divisive:
            self.alpha_initializer = tf.initializers.variance_scaling
        else:
            self.alpha_initializer = tf.ones_initializer
        self.mu_initializer = tf.zeros_initializer
        self.omega_initializer = tf.initializers.variance_scaling
        self.kappa_initializer = tf.zeros_initializer

        # Handle BN scope reuse
        self.scope_reuse = reuse

        # Load weights
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def __call__(self, rgb, constructor=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        self.input = rgb
        self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.gammanet_constructor = constructor
        start_time = time.time()
        X_shape = rgb.get_shape().as_list()
        self.N = X_shape[0]
        self.dtype = rgb.dtype
        self.prepare_tensors(X_shape, allow_resize=False)
        self.create_hidden_states(
            constructor=self.gammanet_constructor,
            shapes=self.layer_shapes,
            recurrent_ff=self.recurrent_ff,
            init=self.hidden_init,
            dtype=self.dtype)
        self.ff_reuse = self.scope_reuse
        for idx in range(self.timesteps):
            self.build(i0=idx)
            self.ff_reuse = tf.AUTO_REUSE

    def build(self, i0):
        # Convert RGB to BGR
        # self.conv1_1 = self.conv_layer(self.input, "conv1_1")
        # self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
            ff_drive=self.conv1_2,
            h2=self.fgru_0,
            layer_id=0,
            i0=i0)
        self.fgru_0 = fgru_activity
        self.pool1 = self.max_pool(self.fgru_0, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
            ff_drive=self.conv5_3,
            h2=self.fgru_1,
            layer_id=1,
            i0=i0)
        self.fgru_1 = fgru_activity

        # Resize and conv
	fgru_0_td = tf.image.resize_nearest_neighbor(self.fgru_1, self.fgru_0.get_shape().as_list()[1:3], align_corners=True)
        fgru_0_td = self.conv_layer(fgru_0_td, '5_to_1', learned=True, shape=[3, 3, fgru_0_td.get_shape().as_list()[-1], 64]) 

        # TD
        error, fgru_activity = self.fgru_ops(  # h^(1), h^(2)
            ff_drive=self.fgru_0,
            h2=fgru_0_td,
            layer_id=2,
            i0=i0)
        # self.fgru_0 += fgru_activity
        self.fgru_0 = fgru_activity

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #def attention_fun(self, at_maps, activations):
    #    return tf.mul(at_maps,activations)

    #def attention_fun(self, at_maps, activations):
    #    return tf.mul(tf.mul(tf.reduce_max(activations,reduction_indices=[1,2])[:,None,None,:],at_maps),activations)

    def conv_layer(self, bottom, name, learned=False, shape=False):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, learned=learned, shape=shape)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, learned=learned, shape=shape)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name, learned=False, shape=None):
        if learned:
            with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
                return tf.get_variable(
                    name='%s_kernel' % name,
                    shape=shape,
                    dtype=self.dtype,
                    initializer=tf.initializers.variance_scaling)                                
        else:
            return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name, learned=False, shape=None):
        if learned:
            with tf.variable_scope('ff_vars', reuse=self.ff_reuse):
                return tf.get_variable(
                    name='%s_bias' % name,
                    shape=[shape[-1]],
                    dtype=self.dtype,
                    initializer=tf.initializers.zeros)
        else:
            return tf.constant(self.data_dict[name][1], name="biases")
