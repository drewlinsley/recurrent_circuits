"""Contextual model with partial filters."""
import numpy as np
import tensorflow as tf
from ops import initialization


# Dependency for symmetric weight ops is in models/layers/ff.py
class hGRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            layer_name,
            x_shape,
            timesteps=1,
            h_ext=15,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            data_format='NHWC',
            train=True):
        """Global initializations and settings."""
        if data_format == 'NHWC':
            self.n, self.h, self.w, self.k = x_shape
            self.bias_shape = [1, 1, 1, self.k]
        elif data_format == 'NCHW':
            self.n, self.k, self.h, self.w = x_shape
            self.bias_shape = [1, self.k, 1, 1]
        else:
            raise NotImplementedError(data_format)
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding
        self.train = train
        self.layer_name = layer_name
        self.data_format = data_format
        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.h_ext = h_ext
        self.h_shape = [self.h_ext, self.h_ext, self.k, self.k]
        self.g_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.m_shape = [self.gate_filter, self.gate_filter, self.k, self.k]

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

        # Set integration operations
        self.ii, self.oi = self.input_integration, self.output_integration

        # Handle BN scope reuse
        if self.reuse:
            self.scope_reuse = tf.AUTO_REUSE
        else:
            self.scope_reuse = None
        self.param_initializer = {
            'moving_mean': tf.constant_initializer(0.),
            'moving_variance': tf.constant_initializer(1.),
            'gamma': tf.constant_initializer(0.1)
        }
        self.param_trainable = {
            'moving_mean': False,
            'moving_variance': False,
            'gamma': True
        }
        self.param_collections = {
            'moving_mean': None,  # [tf.GraphKeys.UPDATE_OPS],
            'moving_variance': None,  # [tf.GraphKeys.UPDATE_OPS],
            'gamma': None
        }
        self.kernel_initializer = tf.glorot_uniform_initializer  # orthogonal  # tf.initializers.variance_scaling

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'lesion_alpha': False,
            'lesion_mu': False,
            'lesion_omega': False,
            'lesion_kappa': False,
            'dtype': tf.float32,
            'hidden_init': 'zeros',
            'gate_bias_init': 'chronos',
            # 'train': True,
            'while_loop': False,
            'recurrent_nl': tf.nn.relu,
            'gate_nl': tf.nn.sigmoid,
            'normal_initializer': False,
            'symmetric_weights': True,
            'symmetric_gate_weights': False,
            'force_divisive': True,
            'symmetric_inits': True,
            'mirror_horizontal': False,
            'nonnegative': True,
            'gate_filter': 1,  # Gate kernel size
            'gamma': True,  # Scale P
            'alpha': True,  # divisive eCRF
            'mu': True,  # subtractive eCRF
            'adaptation': False,
            'multiplicative_excitation': True,
            'horizontal_dilations': [1, 1, 1, 1],
            'reuse': False,
            'constrain': False  # Constrain greek letters to be +
        }

    def interpret_nl(self, nl_type):
        """Return activation function."""
        if nl_type == 'tanh':
            return tf.nn.tanh
        elif nl_type == 'relu':
            return tf.nn.relu
        elif nl_type == 'selu':
            return tf.nn.selu
        elif nl_type == 'leaky_relu':
            return tf.nn.leaky_relu
        elif nl_type == 'sigmoid':
            return tf.sigmoid
        elif nl_type == 'hard_tanh':
            return lambda z: tf.maximum(tf.minimum(z, 1), 0)
        elif nl_type == 'relu6':
            return tf.nn.relu6
        else:
            raise NotImplementedError(nl_type)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def symmetric_initializer(self, w):
        """Initialize symmetric weight sharing."""
        return 0.5 * (w + tf.transpose(w, (0, 1, 3, 2)))

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        if self.constrain:
            constraint = lambda x: tf.clip_by_value(x, 0, np.infty)
        else:
            constraint = None
        self.var_scope = '%s_hgru_weights' % self.layer_name
        recurrent_kernel = [self.h_ext, self.h_ext]
        with tf.variable_scope(self.var_scope):
            iv = self.kernel_initializer()(
                recurrent_kernel + [self.k, self.k])
            ev = self.kernel_initializer()(
                recurrent_kernel + [self.k, self.k])
            if self.symmetric_weights and self.symmetric_inits:
                iv = self.symmetric_initializer(iv)
                ev = self.symmetric_initializer(ev)
            self.horizontal_kernels_inh = tf.get_variable(
                name='%s_horizontal_inh' % self.layer_name,
                dtype=self.dtype,
                initializer=iv,
                trainable=self.train)
            if not self.mirror_horizontal:
                self.horizontal_kernels_exc = tf.get_variable(
                    name='%s_horizontal_exc' % self.layer_name,
                    dtype=self.dtype,
                    initializer=ev,
                    trainable=self.train)
            self.gain_kernels = tf.get_variable(
                name='%s_gain' % self.layer_name,
                dtype=self.dtype,
                shape=self.g_shape,
                initializer=self.kernel_initializer(),  # tf.initializers.variance_scaling(),
                trainable=self.train)
            self.mix_kernels = tf.get_variable(
                name='%s_mix' % self.layer_name,
                dtype=self.dtype,
                shape=self.m_shape,
                initializer=self.kernel_initializer(),  # tf.initializers.variance_scaling(),
                trainable=self.train)
            if self.symmetric_gate_weights and self.symmetric_inits:
                self.gain_kernels = self.symmetric_init(self.gain_kernels)
                self.mix_kernels = self.symmetric_init(self.mix_kernels)

            # BN params
            setattr(self, 'g1_gamma', tf.get_variable(
                name='%s_g1_gamma' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                shape=self.bias_shape,
                initializer=self.param_initializer['gamma']))
            setattr(self, 'g2_gamma', tf.get_variable(
                name='%s_g2_gamma' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                shape=self.bias_shape,
                initializer=self.param_initializer['gamma']))
            setattr(self, 'c1_gamma', tf.get_variable(
                name='%s_c1_gamma' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                shape=self.bias_shape,
                initializer=self.param_initializer['gamma']))
            setattr(self, 'c2_gamma', tf.get_variable(
                name='%s_c2_gamma' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                shape=self.bias_shape,
                initializer=self.param_initializer['gamma']))

            # Gain bias
            if self.gate_bias_init == 'chronos':
                bias_init = -tf.log(
                    tf.random_uniform(
                        self.bias_shape, minval=1, maxval=self.timesteps - 1))
            else:
                bias_init = -tf.ones(self.bias_shape)
            self.gain_bias = tf.get_variable(
                name='%s_gain_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=bias_init)
            if self.gate_bias_init == 'chronos':
                bias_init = -bias_init
            else:
                bias_init = tf.ones(self.bias_shape)
            self.mix_bias = tf.get_variable(
                name='%s_mix_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=self.train,
                initializer=bias_init)

            # Divisive params
            if self.alpha and not self.lesion_alpha:
                self.alpha = tf.get_variable(
                    name='%s_alpha' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.constant_initializer(0.1),
                    trainable=self.train)
            elif self.lesion_alpha:
                self.alpha = tf.constant(0.)
            else:
                self.alpha = tf.constant(1.)

            if self.mu and not self.lesion_mu:
                self.mu = tf.get_variable(
                    name='%s_mu' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.constant_initializer(0.),
                    trainable=self.train)

            elif self.lesion_mu:
                self.mu = tf.constant(0.)
            else:
                self.mu = tf.constant(1.)

            if self.gamma:
                self.gamma = tf.get_variable(
                    name='%s_gamma' % self.layer_name,
                    constraint=constraint,
                    shape=self.bias_shape,
                    initializer=tf.constant_initializer(1.),
                    trainable=self.train)
            else:
                self.gamma = tf.constant(1.)

            if self.multiplicative_excitation:
                if self.lesion_kappa:
                    self.kappa = tf.constant(0.)
                else:
                    self.kappa = tf.get_variable(
                        name='%s_kappa' % self.layer_name,
                        constraint=constraint,
                        shape=self.bias_shape,
                        initializer=tf.constant_initializer(0.5),
                        trainable=self.train)

                if self.lesion_omega:
                    self.omega = tf.constant(0.)
                else:
                    self.omega = tf.get_variable(
                        name='%s_omega' % self.layer_name,
                        constraint=constraint,
                        shape=self.bias_shape,
                        initializer=tf.constant_initializer(0.5),
                        trainable=self.train)
            else:
                self.kappa = tf.constant(1.)
                self.omega = tf.constant(1.)

            if self.adaptation:
                self.eta = tf.get_variable(
                    trainable=self.train,
                    name='%s_eta' % self.layer_name,
                    shape=[self.timesteps],
                    initializer=tf.constant_initializer(1./self.timesteps))
            if self.lesion_omega:
                self.omega = tf.constant(0.)
            if self.lesion_kappa:
                self.kappa = tf.constant(0.)
            if self.reuse:
                # Make the batchnorm variables
                scopes = ['g1_bn', 'g2_bn', 'c1_bn', 'c2_bn']
                bn_vars = ['moving_mean', 'moving_variance', 'gamma']
                for s in scopes:
                    with tf.variable_scope(s) as scope:
                        for v in bn_vars:
                            tf.get_variable(
                                trainable=self.param_trainable[v],
                                name=v,
                                shape=[self.k],
                                collections=self.param_collections[v],
                                initializer=self.param_initializer[v])
                self.param_initializer = None

    def conv_2d_op(
            self,
            data,
            weights,
            dilations=[1, 1, 1, 1],
            symmetric_weights=False):
        """2D convolutions for hgru."""
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            if symmetric_weights:
                g = tf.get_default_graph()
                with g.gradient_override_map({'Conv2D': 'ChannelSymmetricConv'}):
                    activities = tf.nn.conv2d(
                        data,
                        weights,
                        self.strides,
                        dilations=dilations,
                        data_format=self.data_format,
                        padding=self.padding)
            else:
                activities = tf.nn.conv2d(
                    data,
                    weights,
                    self.strides,
                    dilations=dilations,
                    data_format=self.data_format,
                    padding=self.padding)
        else:
            raise RuntimeError
        return activities

    def circuit_input(self, var_scope, h2):
        """Calculate gain and inh horizontal activities."""
        g1_intermediate = self.conv_2d_op(
            data=h2,
            weights=self.gain_kernels,
            symmetric_weights=self.symmetric_gate_weights)
        with tf.variable_scope(
                '%s/g1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g1_intermediate = tf.contrib.layers.batch_norm(
                inputs=g1_intermediate + self.gain_bias,
                scale=False,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
            g1_intermediate *= getattr(self, 'g1_gamma')
        g1 = self.gate_nl(g1_intermediate)

        # Horizontal activities
        c1 = self.conv_2d_op(
            data=h2 * g1,
            weights=self.horizontal_kernels_inh,
            symmetric_weights=self.symmetric_weights,
            dilations=self.horizontal_dilations)
        with tf.variable_scope(
                '%s/c1_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c1 = tf.contrib.layers.batch_norm(
                inputs=c1,
                scale=False,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
            c1 *= getattr(self, 'c1_gamma')
        return c1

    def circuit_output(self, var_scope, h1):
        """Calculate mix and exc horizontal activities."""
        g2_intermediate = self.conv_2d_op(
            data=h1,
            weights=self.mix_kernels,
            symmetric_weights=self.symmetric_gate_weights)
        with tf.variable_scope(
                '%s/g2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            g2_intermediate = tf.contrib.layers.batch_norm(
                inputs=g2_intermediate + self.mix_bias,
                scale=False,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
            g2_intermediate *= getattr(self, 'g2_gamma')
        # Calculate and apply dropout if requested
        g2 = self.gate_nl(g2_intermediate)

        # Horizontal activities
        if self.mirror_horizontal:
            c2 = self.conv_2d_op(
                data=h1,
                weights=-1 * self.horizontal_kernels_inh,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        else:
            c2 = self.conv_2d_op(
                data=h1,
                weights=self.horizontal_kernels_exc,
                symmetric_weights=self.symmetric_weights,
                dilations=self.horizontal_dilations)
        with tf.variable_scope(
                '%s/c2_bn' % var_scope,
                reuse=self.scope_reuse) as scope:
            c2 = tf.contrib.layers.batch_norm(
                inputs=c2,
                scale=False,
                center=False,
                fused=True,
                renorm=False,
                param_initializers=self.param_initializer,
                updates_collections=None,
                scope=scope,
                reuse=self.reuse,
                is_training=self.train)
            c2 *= getattr(self, 'c2_gamma')
        return c2, g2

    def input_integration(self, x, c1, h2):
        """Integration on the input."""
        if self.force_divisive:
            alpha = tf.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        if self.nonnegative:
            return self.recurrent_nl(
                x - self.recurrent_nl((alpha * h2 + self.mu) * c1))
        else:
            return self.recurrent_nl(
                x - ((alpha * h2 + self.mu) * c1))

    def output_integration(self, h1, c2, g2, h2):
        """Integration on the output."""
        if self.multiplicative_excitation:
            # Multiplicative gating I * (P + Q)
            e = self.gamma * c2
            a = self.kappa * (h1 + e)
            m = self.omega * (h1 * e)
            h2_hat = self.recurrent_nl(a + m)
        else:
            # Additive gating I + P + Q
            h2_hat = self.recurrent_nl(
                h1 + self.gamma * c2)
        return (g2 * h2) + ((1 - g2) * h2_hat)

    def full(self, i0, x, h1, h2, var_scope='hgru_weights'):
        """hGRU body."""
        var_scope = 'hgru_weights'
        if not self.while_loop:
            var_scope = '%s_t%s' % (var_scope, i0)

        c1 = self.circuit_input(
            var_scope=var_scope,
            h2=h2)

        # Calculate input (-) integration: h1 (4)
        h1 = self.input_integration(
            x=x,
            c1=c1,
            h2=h2)

        # Circuit output receives recurrent input h1
        c2, g2 = self.circuit_output(var_scope=var_scope, h1=h1)

        # Calculate output (+) integration: h2 (8, 9)
        h2 = self.output_integration(
            h1=h1,
            c2=c2,
            g2=g2,
            h2=h2)
        if self.adaptation:
            e = tf.gather(self.eta, i0, axis=-1)
            h2 *= e

        # Iterate loop
        i0 += 1
        return i0, x, h1, h2

    def condition(self, i0, x, h1, h2):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        x_shape = x.get_shape().as_list()
        if self.hidden_init == 'identity':
            h1 = tf.identity(x)
            h2 = tf.identity(x)
        elif self.hidden_init == 'random':
            h1 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
            h2 = initialization.xavier_initializer(
                shape=x_shape,
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            h1 = tf.zeros_like(x)
            h2 = tf.zeros_like(x)
        else:
            raise RuntimeError

        if not self.while_loop:
            for idx in range(self.timesteps):
                _, x, h1, h2 = self.full(
                    i0=idx,
                    x=x,
                    h1=h1,
                    h2=h2)
        else:
            # While loop
            i0 = tf.constant(0)
            elems = [
                i0,
                x,
                h1,
                h2
            ]
            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            i0, x, h1, h2 = returned
        return h2
