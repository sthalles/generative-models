import tensorflow as tf


class ConditionalBatchNorm(tf.keras.layers.Layer):
    """Conditional BatchNorm.
    For each  class, it has a specific gamma and beta as normalization variable.
    """

    def __init__(self, num_categories, decay_rate=0.999, center=True, scale=True, variance_epsilon=1e-5):
        super(ConditionalBatchNorm, self).__init__()
        self.num_categories = num_categories
        self.center = center
        self.scale = scale
        self.decay_rate = decay_rate
        self.moving_mean = None
        self.moving_var = None
        self.variance_epsilon = variance_epsilon

    def build(self, input_shape):
        params_shape = input_shape[-1:]
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)
        moving_shape = tf.TensorShape([1, 1, 1]).concatenate(params_shape)

        self.gamma = self.add_weight(
            'gamma', shape,
            initializer=tf.ones_initializer())

        self.beta = self.add_weight(
            'beta', shape,
            initializer=tf.zeros_initializer())

        self.moving_mean = self.add_weight('mean', moving_shape,
                                           initializer=tf.zeros_initializer(),
                                           trainable=False)

        self.moving_var = self.add_weight('var', moving_shape,
                                          initializer=tf.ones_initializer(),
                                          trainable=False)

    def call(self, x, y, training):
        y = tf.squeeze(tf.cast(y, tf.int32))
        x = tf.convert_to_tensor(x)
        inputs_shape = x.get_shape()
        axis = [0, 1, 2]

        beta = tf.gather(self.beta, y)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = tf.gather(self.gamma, y)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        decay = self.decay_rate

        if training:
            mean, variance = tf.nn.moments(x, axis, keepdims=True)
            self.moving_mean.assign(self.moving_mean * decay + mean * (1 - decay))
            self.moving_var.assign(self.moving_var * decay + variance * (1 - decay))

            outputs = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, self.variance_epsilon)
        else:
            outputs = tf.nn.batch_normalization(
                x, self.moving_mean, self.moving_var, beta, gamma, self.variance_epsilon)
        outputs.set_shape(inputs_shape)
        return outputs
