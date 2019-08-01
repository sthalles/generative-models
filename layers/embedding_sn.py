import tensorflow as tf
from layers.ops import power_iteration


class SNEmbeeding(tf.keras.Model):
    def __init__(self, embedding_size, n_classes, kernel_initializer='normal', max_iters=1, kernel_regularizer=None):
        super(SNEmbeeding, self).__init__()
        self.max_iters = max_iters
        self.n_classes = n_classes
        self.embedding_size = embedding_size
        self.kernel_initializer=kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):

        self.embedding_map = self.add_weight(shape=(self.n_classes, self.embedding_size),
                                             initializer=self.kernel_initializer,
                                             trainable=True,
                                             regularizer=self.kernel_regularizer,
                                             name='embedding_map',
                                             dtype=tf.float32)

        self.u = self.add_weight(
            'sn_estimate',
            shape=[1, self.n_classes],
            initializer='normal',
            dtype=self.embedding_map.dtype,
            trainable=False)

        assert (self.u.shape == (1, self.n_classes))

    def compute_spectral_normal(self, weights, sn_update):
        # Spectrally Normalized Weight
        W_shape = weights.shape.as_list()

        # Flatten the Tensor
        W_mat = tf.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]
        W_sn, u, v = power_iteration(W_mat, self.u, rounds=self.max_iters)

        if sn_update == True:
            # Update estimated 1st singular vector
            self.u.assign(u)

        W_mat = W_mat / W_sn
        w_bar = tf.reshape(W_mat, W_shape)

        return w_bar

    def call(self, x, sn_update):
        assert sn_update is not None, "sn_update parameter not provided."
        embedding_map_bar_transpose = self.compute_spectral_normal(tf.transpose(self.embedding_map), sn_update=sn_update)
        embedding_map_bar = tf.transpose(embedding_map_bar_transpose)
        return tf.nn.embedding_lookup(embedding_map_bar, x)
