import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def normalize_images(images):
    images = images.astype("float32") / 255.0
    return images

def flatten_images(images):
    images = images.reshape(-1, 28 * 28)
    return images

def unflatten_images(vectors):
    images = vectors.reshape(-1, 28, 28)
    return images

def select_subset(X, y, n_samples, seed):
    # convert to tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # select subset
    indices = tf.random.shuffle(tf.range(tf.shape(X)[0]), seed=seed)[:n_samples]
    X_subset = tf.gather(X, indices)
    y_subset = tf.gather(y, indices)

    return X_subset, y_subset
