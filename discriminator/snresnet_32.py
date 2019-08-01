import tensorflow as tf
from layers.conv_sn import SNConv2D
from layers.dense_sn import SNDense
from layers.embedding_sn import SNEmbeeding
from discriminator.resblocks import OptimizedBlock, Block
from layers.sn_non_local_block import SNNonLocalBlock

class SNResNetProjectionDiscriminator(tf.keras.Model):
    def __init__(self, ch=64, n_classes=0, activation=tf.keras.layers.ReLU()):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = tf.keras.initializers.glorot_uniform()
        self.block1 = OptimizedBlock(ch)
        self.block2 = Block(ch, ch, activation=activation, downsample=True)
        self.self_atten = SNNonLocalBlock(ch)
        self.block3 = Block(ch, ch, activation=activation, downsample=False)
        self.block4 = Block(ch, ch, activation=activation, downsample=False)
        self.l6 = SNDense(units=1, kernel_initializer=initializer)
        if n_classes > 0:
            self.l_y = SNEmbeeding(embedding_size=ch, n_classes=n_classes, kernel_initializer=initializer)

    def __call__(self, x, y=None, sn_update=None):
        assert sn_update is not None, "Define the 'sn_update' parameter"
        h = x
        h = self.block1(h, sn_update=sn_update)
        h = self.block2(h, sn_update=sn_update)
        h = self.self_atten(h, sn_update=sn_update)
        h = self.block3(h, sn_update=sn_update)
        h = self.block4(h, sn_update=sn_update)
        h = self.activation(h)
        h = tf.reduce_sum(h, axis=(1, 2))  # Global pooling
        output = self.l6(h, sn_update=sn_update)
        if y is not None:
            w_y = self.l_y(y, sn_update=sn_update)
            output += tf.reduce_sum(w_y * h, axis=1, keepdims=True)
        return output