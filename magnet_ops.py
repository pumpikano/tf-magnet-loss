import tensorflow as tf
import numpy as np


def magnet_loss(r, classes, clusters, cluster_classes, n_clusters, alpha=1.0):
    """Compute magnet loss.

    Given a tensor of features `r`, the assigned class for each example,
    the assigned cluster for each example, the assigned class for each 
    cluster, the total number of clusters, and separation hyperparameter,
    compute the magnet loss according to equation (4) in
    http://arxiv.org/pdf/1511.05939v2.pdf.

    Note that cluster and class indexes should be sequential startined at 0.

    Args:
        r: A batch of features.
        classes: Class labels for each example.
        clusters: Cluster labels for each example.
        cluster_classes: Class label for each cluster.
        n_clusters: Total number of clusters.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """
    # Helper to compute boolean mask for distance comparisons
    def comparison_mask(a_labels, b_labels):
        return tf.equal(tf.expand_dims(a_labels, 1),
                        tf.expand_dims(b_labels, 0))

    # Take cluster means within the batch
    cluster_examples = tf.dynamic_partition(r, clusters, n_clusters)
    cluster_means = tf.stack([tf.reduce_mean(x, 0) for x in cluster_examples])

    # Compute squared distance of each example to each cluster centroid
    sample_costs = tf.squared_difference(cluster_means, tf.expand_dims(r, 1))
    sample_costs = tf.reduce_sum(sample_costs, 2)

    # Select distances of examples to their own centroid
    intra_cluster_mask = tf.to_float(comparison_mask(clusters, np.arange(n_clusters, dtype=np.int32)))
    intra_cluster_costs = tf.reduce_sum(intra_cluster_mask * sample_costs, 1)

    # Compute variance of intra-cluster distances
    N = tf.shape(r)[0]
    variance = tf.reduce_sum(intra_cluster_costs) / tf.to_float(N - 1)
    var_normalizer = -1 / (2 * variance**2)

    # Compute numerator
    numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)

    # Compute denominator
    diff_class_mask = tf.to_float(tf.logical_not(comparison_mask(classes, cluster_classes)))
    denom_sample_costs = tf.exp(var_normalizer * sample_costs)
    denominator = tf.reduce_sum(diff_class_mask * denom_sample_costs, 1)

    # Compute example losses and total loss
    epsilon = 1e-8
    losses = tf.nn.relu(-tf.log(numerator / (denominator + epsilon) + epsilon))
    total_loss = tf.reduce_mean(losses)

    return total_loss, losses


def minibatch_magnet_loss(r, classes, m, d, alpha=1.0):
    """Compute minibatch magnet loss.

    Given a batch of features `r` consisting of `m` clusters each with `d`
    consecutive examples, the corresponding class labels for each example
    and a cluster separation gap `alpha`, compute the total magnet loss and
    the per example losses. This is a thin wrapper around `magnet_loss`
    that assumes this particular batch structure to implement minibatch
    magnet loss according to equation (5) in
    http://arxiv.org/pdf/1511.05939v2.pdf. The correct stochastic approximation
    should also follow the cluster sampling procedure presented in section
    3.2 of the paper.

    Args:
        r: A batch of features.
        classes: Class labels for each example.
        m: The number of clusters in the batch.
        d: The number of examples in each cluster.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """
    clusters = np.repeat(np.arange(m, dtype=np.int32), d)
    cluster_classes = tf.strided_slice(classes, [0], [m*d], [d])
    
    return magnet_loss(r, classes, clusters, cluster_classes, m, alpha)
