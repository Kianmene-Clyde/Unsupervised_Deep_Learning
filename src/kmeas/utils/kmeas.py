"""Vectorised and deterministic K-means primitives."""

from __future__ import annotations

import math

import tensorflow as tf

from utils.neural_config import positive_integer


def initialize_centroids(X, k, seed):
    """Choose ``k`` distinct observations with a repeatable local RNG."""
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    n_samples = tf.shape(X)[0]
    indices = tf.random.experimental.stateless_shuffle(
        tf.range(n_samples), seed=tf.constant([int(seed), 0], dtype=tf.int32)
    )[:k]
    return tf.gather(X, indices)


def compute_squared_distances(X, centroids):
    """Return all squared Euclidean distances with only an ``(n, k)`` buffer."""
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)
    x_squared = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
    c_squared = tf.reduce_sum(tf.square(centroids), axis=1)
    cross = tf.matmul(X, centroids, transpose_b=True)
    # Round-off can make a theoretically null distance slightly negative.
    return tf.maximum(x_squared - 2.0 * cross + c_squared, 0.0)


def assign_clusters(distances):
    return tf.argmin(distances, axis=1, output_type=tf.int32)


def update_centroids(X, labels, k, old_centroids, seed=None):
    """Compute all cluster means in one pass; preserve empty clusters."""
    del seed  # Kept for backward compatibility with the original signature.
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    old_centroids = tf.cast(tf.convert_to_tensor(old_centroids), tf.float32)

    sums = tf.math.unsorted_segment_sum(X, labels, num_segments=k)
    counts = tf.math.unsorted_segment_sum(
        tf.ones((tf.shape(X)[0], 1), dtype=X.dtype), labels, num_segments=k
    )
    means = tf.math.divide_no_nan(sums, counts)
    return tf.where(counts > 0, means, old_centroids)


def compute_inertia(X, labels, centroids):
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)
    distances = compute_squared_distances(X, centroids)
    indices = tf.stack(
        [tf.range(tf.shape(X)[0], dtype=tf.int32), labels], axis=1
    )
    return tf.reduce_sum(tf.gather_nd(distances, indices))


def has_converged(old_centroids, new_centroids, tolerance):
    """Whether every centroid moved by at most ``tolerance`` in L2 norm."""
    old_centroids = tf.cast(tf.convert_to_tensor(old_centroids), tf.float32)
    new_centroids = tf.cast(tf.convert_to_tensor(new_centroids), tf.float32)
    tolerance = tf.cast(tolerance, tf.float32)
    return tf.reduce_all(tf.norm(new_centroids - old_centroids, axis=1) <= tolerance)


def _validated_training_data(X, k, max_iter, tolerance):
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    if X.shape.rank != 2:
        raise ValueError(f"X doit être une matrice 2D, forme reçue : {X.shape}.")
    n_samples = int(tf.shape(X)[0])
    if n_samples == 0 or int(tf.shape(X)[1]) == 0:
        raise ValueError("X doit contenir au moins une observation et une caractéristique.")
    if not bool(tf.reduce_all(tf.math.is_finite(X))):
        raise ValueError("X contient des valeurs NaN ou infinies.")
    positive_integer(k, "k")
    positive_integer(max_iter, "max_iter")
    if not 1 <= int(k) <= n_samples:
        raise ValueError(f"k doit être compris entre 1 et {n_samples}, reçu : {k}.")
    if int(max_iter) < 1:
        raise ValueError("max_iter doit être supérieur ou égal à 1.")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ValueError("tolerance doit être finie et positive ou nulle.")
    return X


def fit_kmeans(X, k, max_iter, tolerance, seed, verbose, return_history=False):
    """Train K-means and return centroids plus labels coherent with them."""
    X = _validated_training_data(X, k, max_iter, tolerance)
    centroids = initialize_centroids(X, int(k), int(seed))
    history = []

    for iteration in range(int(max_iter)):
        labels = assign_clusters(compute_squared_distances(X, centroids))
        new_centroids = update_centroids(X, labels, int(k), centroids)
        inertia = compute_inertia(X, labels, new_centroids)
        history.append(float(inertia))

        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: inertia = {float(inertia):.4f}")

        converged = bool(has_converged(centroids, new_centroids, tolerance))
        centroids = new_centroids
        if converged:
            if verbose:
                print(f"Convergence après {iteration + 1} itérations")
            break

    # Labels were computed against the centroids from the start of the final
    # iteration. Reassign once to make the returned state internally coherent,
    # especially when the iteration cap was reached before convergence.
    labels = assign_clusters(compute_squared_distances(X, centroids))
    history[-1] = float(compute_inertia(X, labels, centroids))

    if return_history:
        return centroids, labels, history
    return centroids, labels
