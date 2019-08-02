import tensorflow as tf
import matplotlib.pyplot as plt
from generator.snresnet_64 import ResNetGenerator
import time
import os
from source.rgb_preprocessing import *
import numpy as np
import yaml

# Read YAML file
with open("./config/cats_and_dogs.yml", 'r') as stream:
    meta_parameters = yaml.safe_load(stream)

IMG_SIZE = meta_parameters['dataset']['img_size']
NUM_CLASSES = meta_parameters['n_classes']
BATCH_SIZE = 28

gen_parameters = meta_parameters['model']['generator']['args']

Z_DIM = gen_parameters['dim_z']

generator = ResNetGenerator(**gen_parameters)

basefolder = os.path.join("records", str(time.time()))
summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

retrain_from = '1564628105.845846'
checkpoint_dir = './records/' + retrain_from + '/checkpoints'


checkpoint = tf.train.Checkpoint(generator=generator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

# load checkpoints to continue training
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

kwargs = {'training': False}


def generate_fake_batch(batch_size, num_classes, zdim, truncation=1, class_=None):
    z_dim = truncation * tf.random.normal([batch_size, zdim])

    if class_ is None:
        gen_class_logits = tf.zeros((batch_size, num_classes))
        gen_class_ints = tf.random.categorical(gen_class_logits, 1)
        y_fake = tf.squeeze(gen_class_ints)
    else:
        y_fake = tf.constant(class_, shape=[BATCH_SIZE])
    return z_dim, y_fake


def run_generator(sn_update):
    z_dim, y_fake = generate_fake_batch(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, zdim=Z_DIM)

    # tf.summary.histogram(name="z_dim", data=z_dim, step=gen_optimizer.iterations)
    # tf.summary.histogram(name="y_fake", data=y_fake, step=gen_optimizer.iterations)

    x_fake = generator(z=z_dim, y=y_fake, sn_update=sn_update, **kwargs)
    return x_fake, y_fake

truncation = meta_parameters['truncation']  # scalar truncation value in [0.0, 1.0]


def generate_samples():

    for _ in range(8):

        fig, axs = plt.subplots(nrows=4, ncols=7, constrained_layout=False)

        c = np.random.choice(list(range(NUM_CLASSES)))

        # sample a fake batch
        z_dim, y_fake = generate_fake_batch(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, zdim=Z_DIM, truncation=truncation, class_=c)

        x_fake = generator(z=z_dim, y=y_fake, sn_update=False, **kwargs)
        for i, ax in enumerate(axs.flat):
            ax.imshow((x_fake[i]+1) * 0.5)

        plt.show()

generate_samples()
