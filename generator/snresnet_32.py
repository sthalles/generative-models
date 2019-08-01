import tensorflow as tf
from generator.resblocks import Block
from layers.dense_sn import SNDense
from layers.conv_sn import SNConv2D
from layers.sn_non_local_block import SNNonLocalBlock

class ResNetGenerator(tf.keras.Model):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=tf.keras.layers.ReLU(),
                 n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = tf.keras.initializers.glorot_uniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes

        self.l1 = SNDense(units=(bottom_width ** 2) * ch * 16, kernel_initializer=initializer)
        self.block2 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.self_atten = SNNonLocalBlock(ch)
        self.block3 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b6 = tf.keras.layers.BatchNormalization()
        self.l6 = SNConv2D(3, kernel_size=3, strides=1, padding="SAME", kernel_initializer=initializer)

    def __call__(self, z=None, y=None, sn_update=None, **kwargs):

        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h, sn_update=sn_update)
        h = tf.reshape(h, (h.shape[0], self.bottom_width, self.bottom_width, -1))
        h = self.block2(h, y, sn_update=sn_update, **kwargs)
        h = self.self_atten(h, sn_update=sn_update)
        h = self.block3(h, y, sn_update=sn_update, **kwargs)
        h = self.block4(h, y, sn_update=sn_update, **kwargs)
        h = self.b6(h, **kwargs)
        h = self.activation(h)
        h = tf.nn.tanh(self.l6(h, sn_update=sn_update))
        return h