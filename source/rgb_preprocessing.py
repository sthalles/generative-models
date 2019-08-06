import tensorflow as tf


def random_resize(image):
    # Random resize an image
    # For image of original size of 384x384
    # The output can have a maximum height and/or width of [461]
    # and minimum height and/or width of 307
    H, W = image.shape[:2]
    scale = tf.random.uniform([], minval=0.9, maxval=1.1, dtype=tf.float32, seed=None, name=None)
    shape = tf.stack((scale * W, scale * H), axis=0)
    shape = tf.cast(shape, tf.int32)
    image = tf.image.resize(image, size=shape)
    return image


def normalize_rgb(image, target):
    image = image * (2. / 255) - 1.
    image += tf.random.uniform(shape=image.shape, minval=0., maxval=1. / 128)
    return image, target

# Function for resizing
def resize_with_aspect_ratio(image, MAX_SIZE):
    H, W = image.shape[0:2]

    # Take the greater value, and use it for the ratio
    max_ = tf.math.minimum(H, W)
    ratio = tf.cast(max_, tf.float32) / MAX_SIZE

    W_ = tf.cast(W, tf.float32) / ratio
    H_ = tf.cast(H, tf.float32) / ratio

    W_, H_ = tf.cast(W_, tf.int32), tf.cast(H_, tf.int32)
    return tf.image.resize(image, size=(H_, W_))


def resize_with_aspect_ratio(image, MAX_SIZE):
    H, W = image.shape[0:2]

    # Take the greater value, and use it for the ratio
    max_ = tf.math.minimum(H, W)
    ratio = tf.cast(max_, tf.float32) / MAX_SIZE

    W_ = tf.cast(W, tf.float32) / ratio
    H_ = tf.cast(H, tf.float32) / ratio

    W_, H_ = tf.cast(W_, tf.int32), tf.cast(H_, tf.int32)
    image = tf.image.resize(image, size=(H_, W_))
    image = tf.image.resize_with_crop_or_pad(image, target_height=MAX_SIZE, target_width=MAX_SIZE)
    return image


def process_tfds(features, HEIGHT, WIDTH):
    image = features["image"]
    target = features["label"]
    image = tf.image.resize_with_crop_or_pad(image, target_height=32, target_width=32)
    return tf.cast(image, tf.float32), tf.squeeze(target)


def random_crop(image, HEIGHT, WIDTH, CHANNELS=3):
    image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, CHANNELS])
    return image


def random_flip(image, target):
    return tf.image.random_flip_left_right(image), target


def tf_record_parser(record):
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'target': tf.io.FixedLenFeature((), tf.int64)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    image = tf.io.decode_raw(features['image_raw'], tf.float32)
    label = tf.cast(features['target'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (128, 128, 3), name="image_reshape")

    return image, tf.cast(label, tf.int32)