import tensorflow as tf
from discriminator.snresnet_64 import SNResNetProjectionDiscriminator
from generator.snresnet_64 import ResNetGenerator

# from discriminator.snresnet_32 import SNResNetProjectionDiscriminator
# from generator.snresnet_32 import ResNetGenerator
from source.loss import loss_hinge_dis, loss_hinge_gen
import time
import os
from distutils.dir_util import copy_tree
import tensorflow_datasets as tfds
from shutil import copyfile
from source.rgb_preprocessing import *
import numpy as np
import yaml

# Read YAML file
with open("./config/cats_and_dogs.yml", 'r') as stream:
    meta_parameters = yaml.safe_load(stream)

DISC_UPDATE = meta_parameters['disc_update']
IMG_SIZE = meta_parameters['dataset']['img_size']
BATCH_SIZE = meta_parameters['batchsize']
BUFFER_SIZE = 4096
EPOCHS = meta_parameters['epochs']
NUM_CLASSES = meta_parameters['n_classes']
SUMMARY_EVERY_N_STEPS = meta_parameters['summary_every_n_steps']
SAVE_EVERY_N_STEPS = meta_parameters['save_every_n_steps']

gen_parameters = meta_parameters['model']['generator']['args']
disc_parameters = meta_parameters['model']['discriminator']['args']

Z_DIM = gen_parameters['dim_z']

train_dataset = tf.data.TFRecordDataset('./tfrecords/train.tfrecords')
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.map(random_flip)
train_dataset = train_dataset.map(normalize_rgb)
# train_dataset = train_dataset.map(batch_gan)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.repeat(EPOCHS)
train_dataset = train_dataset.batch(BATCH_SIZE)

basefolder = os.path.join("records", str(time.time()))

generator = ResNetGenerator(**gen_parameters)
discriminator = SNResNetProjectionDiscriminator(**disc_parameters)

gen_optimizer_args = meta_parameters['optimizer']['generator']
gen_optimizer = tf.keras.optimizers.Adam(**gen_optimizer_args)

dis_optimizer_args = meta_parameters['optimizer']['discriminator']
dis_optimizer = tf.keras.optimizers.Adam(**dis_optimizer_args)

summary_path = os.path.join(basefolder, 'summary')
train_summary_writer = tf.summary.create_file_writer(summary_path)

copy_tree('./generator', os.path.join(basefolder, 'generator'))
copy_tree('./discriminator', os.path.join(basefolder, 'discriminator'))
copyfile('./train.py', os.path.join(basefolder, 'train.py'))

retrain_from = '1564628105.845846'
if retrain_from is not None:
    checkpoint_dir = './records/' + retrain_from + '/checkpoints'
else:
    checkpoint_dir = os.path.join(basefolder, 'checkpoints')

checkpoint = tf.train.Checkpoint(generator=generator,
                                 gen_optimizer=gen_optimizer,
                                 dis_optimizer=dis_optimizer,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)

# load checkpoints to continue training
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

kwargs = {'training': True}


def generate_fake_batch(batch_size, num_classes, zdim, truncation=1):
    z_dim = truncation * tf.random.normal([batch_size, zdim])

    gen_class_logits = tf.zeros((batch_size, num_classes))
    gen_class_ints = tf.random.categorical(gen_class_logits, 1)
    y_fake = tf.squeeze(gen_class_ints)
    return z_dim, y_fake


def run_generator(sn_update):
    z_dim, y_fake = generate_fake_batch(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, zdim=Z_DIM)

    # tf.summary.histogram(name="z_dim", data=z_dim, step=gen_optimizer.iterations)
    # tf.summary.histogram(name="y_fake", data=y_fake, step=gen_optimizer.iterations)

    x_fake = generator(z=z_dim, y=y_fake, sn_update=sn_update, **kwargs)
    return x_fake, y_fake


@tf.function
def generator_train_step():
    with tf.GradientTape() as gen_tape:
        x_fake, y_fake = run_generator(sn_update=True)
        disc_fake = discriminator(x=x_fake, y=y_fake, sn_update=True)

        regularization_loss = tf.math.add_n(generator.losses)
        gen_loss = loss_hinge_gen(dis_fake=disc_fake)
        gen_loss += regularization_loss

        tf.summary.scalar('generator_loss', gen_loss, step=gen_optimizer.iterations)
        tf.summary.scalar('regularization_loss', regularization_loss, step=gen_optimizer.iterations)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(generator_gradients,
                                      generator.trainable_variables))


@tf.function
def discriminator_train_step(x_real, y_real):
    with tf.GradientTape() as disc_tape:
        disc_real = discriminator(x=x_real, y=y_real, sn_update=True)

        x_fake, y_fake = run_generator(sn_update=True)

        disc_fake = discriminator(x=x_fake, y=y_fake, sn_update=True)

        disc_loss = loss_hinge_dis(dis_fake=disc_fake, dis_real=disc_real)

    tf.summary.scalar('discriminator_loss', disc_loss, step=dis_optimizer.iterations)

    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    dis_optimizer.apply_gradients(zip(discriminator_gradients,
                                      discriminator.trainable_variables))


truncation = meta_parameters['truncation']  # scalar truncation value in [0.0, 1.0]


def train():
    with train_summary_writer.as_default():

        for i, (x_real, y_real) in enumerate(train_dataset):

            if i % DISC_UPDATE == 0:
                generator_train_step()

            discriminator_train_step(x_real=x_real, y_real=y_real)

            if tf.math.equal(gen_optimizer.iterations % SUMMARY_EVERY_N_STEPS, 0):
                # sample a fake batch
                z_dim, y_fake = generate_fake_batch(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, zdim=Z_DIM, truncation=truncation)

                x_fake = generator(z=z_dim, y=y_fake, sn_update=False, **kwargs)

                tf.summary.image('generator_image', (x_fake + 1) * 0.5, max_outputs=12, step=gen_optimizer.iterations)
                tf.summary.image('input_images', (x_real + 1) * 0.5, max_outputs=12, step=gen_optimizer.iterations)

            if tf.math.equal(gen_optimizer.iterations % SAVE_EVERY_N_STEPS, 0):
                manager.save()

        print("New checkpoints saved.")


train()
