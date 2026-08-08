import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from .kmeas import compute_squared_distances

def encode(X, centroids):
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)

    # Si X représente une seule image : (784,) -> (1, 784)
    if len(X.shape) == 1:
        X = X[tf.newaxis, :]

    # Shape : (1, K)
    distances = compute_squared_distances(X, centroids)

    # Shape : (1,) ; ex. [4]
    code = tf.argmin(
        distances,
        axis=1,
        output_type=tf.int32
    )

    # On retourne un scalaire ; ex. 4
    return code[0]
    
def decode(code, centroids):
    code = tf.cast(code, tf.int32)
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)

    # Ex. code = 4 -> retourne centroids[4], shape (784,)
    centroid = tf.gather(centroids, code)

    return centroid
