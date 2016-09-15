def magnet_loss_old(r, m, d, alpha=1.0):
    """Compute magnet loss for batch.
    
    Given a batch of features r consisting of m batches
    each with d assigned examples and a cluster separation
    gap of alpha, compute the total magnet loss and the per
    example losses.
    
    Args:
        r: A batch of features.
        m: The number of clusters in the batch.
        d: The number of examples in each cluster.
        alpha: The cluster separation gap hyperparameter.
        
    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    
    """

    # Take cluster means within the batch
    cluster_means = tf.reduce_mean(tf.reshape(r, [m, d, -1]), 1)

    # Compute squared differences of each example to each cluster centroid
    sample_cluster_pair_inds = np.array(list(product(range(m*d), range(m))))
    sample_costs = tf.squared_difference(
        tf.gather(r, sample_cluster_pair_inds[:,0]),
        tf.gather(cluster_means, sample_cluster_pair_inds[:,1]))

    # Sum to compute squared distances of each example to each cluster centroid
    # and reshape such that tensor is indexed by
    # [true cluster, comparison cluster, example in true cluster]
    sample_costs = tf.reshape(tf.reduce_sum(sample_costs, 1), [m, d, m])
    sample_costs = tf.transpose(sample_costs, [0, 2, 1])

    # Select distances of examples to their own centroid
    same_cluster_inds = np.vstack(np.diag_indices(m)).T
    intra_cluster_costs = tf.gather_nd(sample_costs, same_cluster_inds)

    # Select distances of examples to other centroids and reshape such that
    # tensor is indexed by [true cluster, comparison cluster, example]
    cluster_inds = np.arange(m)
    diff_cluster_inds = np.vstack(
        [np.repeat(cluster_inds, m-1), 
         np.hstack([cluster_inds[cluster_inds != i] for i in range(m)])]).T
    inter_cluster_costs = tf.reshape(tf.gather_nd(sample_costs, diff_cluster_inds), [m, m-1, d])

    # Compute variance of intra-cluster squared distances
    variance = tf.reduce_sum(intra_cluster_costs) / (m * d - 1)
    var_normalizer = -1 / 2*variance**2

    # Compute numerator and denominator of inner term
    numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)
    denominator = tf.reduce_sum(tf.exp(var_normalizer * inter_cluster_costs), 1)

    # Compute example losses and total loss
    losses = tf.nn.relu(-tf.log(numerator / denominator))
    total_loss = tf.reduce_mean(losses)
    
    return total_loss, losses


def magnet_loss(r, m, d, alpha=1.0):
    """Compute magnet loss for batch.
    
    Given a batch of features r consisting of m batches
    each with d assigned examples and a cluster separation
    gap of alpha, compute the total magnet loss and the per
    example losses.
    
    Args:
        r: A batch of features.
        m: The number of clusters in the batch.
        d: The number of examples in each cluster.
        alpha: The cluster separation gap hyperparameter.
        
    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    
    """
    
    # Helper to compute indexes to select intra- and inter-cluster
    # distances
    def compute_comparison_inds():
        same_cluster_inds = []
        for i in range(m*d):
            c = i / d
            same_cluster_inds.append(c*d*m + c*d + (i % d))
        diff_cluster_inds = sorted(set(range(m*m*d)) - set(same_cluster_inds))
        
        return same_cluster_inds, diff_cluster_inds

        
    # Take cluster means within the batch
    cluster_means = tf.reduce_mean(tf.reshape(r, [m, d, -1]), 1)

    # Compute squared distance of each example to each cluster centroid
    sample_cluster_pair_inds = np.array(list(product(range(m), range(m*d))))
    sample_costs = tf.squared_difference(
        tf.gather(cluster_means, sample_cluster_pair_inds[:,0]),
        tf.gather(r, sample_cluster_pair_inds[:,1]))
    sample_costs = tf.reduce_sum(sample_costs, 1)
    
    # Compute intra- and inter-cluster comparison indexes
    same_cluster_inds, diff_cluster_inds = compute_comparison_inds()
    
    # Select distances of examples to their own centroid
    intra_cluster_costs = tf.gather(sample_costs, same_cluster_inds)
    intra_cluster_costs = tf.reshape(intra_cluster_costs, [m, d])
    
    # Select distances of examples to other centroids
    inter_cluster_costs = tf.reshape(tf.gather(sample_costs, diff_cluster_inds), [m, m-1, d])

    # Compute variance of intra-cluster squared distances
    variance = tf.reduce_sum(intra_cluster_costs) / (m * d - 1)
    var_normalizer = -1 / 2*variance**2

    # Compute numerator and denominator of inner term
    numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)
    denominator = tf.reduce_sum(tf.exp(var_normalizer * inter_cluster_costs), 1)

    # Compute example losses and total loss
    losses = tf.nn.relu(-tf.log(numerator / denominator))
    total_loss = tf.reduce_mean(losses)
    
    return total_loss, losses



def magnet_loss3(r, m, d, alpha=1.0):
    """Compute magnet loss for batch.
    
    Given a batch of features r consisting of m batches
    each with d assigned examples and a cluster separation
    gap of alpha, compute the total magnet loss and the per
    example losses.
    
    Args:
        r: A batch of features.
        m: The number of clusters in the batch.
        d: The number of examples in each cluster.
        alpha: The cluster separation gap hyperparameter.
        
    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    
    """
    
    # Helper to compute indexes to select intra- and inter-cluster
    # distances
    def compute_comparison_inds():
        same_cluster_inds = []
        for i in range(m*d):
            c = i / d
            same_cluster_inds.append(c*d*m + c*d + (i % d))
        diff_cluster_inds = sorted(set(range(m*m*d)) - set(same_cluster_inds))
        
        return same_cluster_inds, diff_cluster_inds

        
    # Take cluster means within the batch
    cluster_means = tf.reduce_mean(tf.reshape(r, [m, d, -1]), 1)

    # Compute squared distance of each example to each cluster centroid
    sample_cluster_pair_inds = np.array(list(product(range(m), range(m*d))))
    sample_costs = tf.squared_difference(
        tf.gather(cluster_means, sample_cluster_pair_inds[:,0]),
        tf.gather(r, sample_cluster_pair_inds[:,1]))
    sample_costs = tf.reduce_sum(sample_costs, 1)
    
    # Compute intra- and inter-cluster comparison indexes
    same_cluster_inds, diff_cluster_inds = compute_comparison_inds()
    
    # Select distances of examples to their own centroid
    intra_cluster_costs = tf.gather(sample_costs, same_cluster_inds)
    intra_cluster_costs = tf.reshape(intra_cluster_costs, [m, d])
    
    # Select distances of examples to other centroids
    inter_cluster_costs = tf.reshape(tf.gather(sample_costs, diff_cluster_inds), [m, m-1, d])

    # Compute variance of intra-cluster squared distances
    variance = tf.reduce_sum(intra_cluster_costs) / (m * d - 1)
    var_normalizer = -1 / 2*variance**2

    # Compute numerator and denominator of inner term
    numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)
    denominator = tf.reduce_sum(tf.exp(var_normalizer * inter_cluster_costs), 1)

    # Compute example losses and total loss
    losses = tf.nn.relu(-tf.log(numerator / denominator))
    total_loss = tf.reduce_mean(losses)
    
    return total_loss, losses


    
sess = tf.InteractiveSession()

m = 3
d = 2

m = 6
d = 4

K = 5
alpha = 15.0

r = tf.placeholder(tf.float32, [None, 8])
magnet_loss1, losses1 = magnet_loss_old(r, m, d, alpha)
magnet_loss2, losses2 = magnet_loss(r, m, d, alpha)


# Helper to generate debug data
def gen_data(m, d):
    data = []
    for c in range(m):
        a = (c + 1) * 3
        centroid = np.random.random([1, 8]) * a
        data.append(centroid + np.random.random([d, 8]))
    return np.vstack(data)


# feed_dict = {r: np.random.random([m*d, 8])}
feed_dict = {r: gen_data(m, d)}

print sess.run([magnet_loss1, magnet_loss2], feed_dict=feed_dict)


sess.close()
tf.reset_default_graph()





# # trying to simplify cost shape
# def magnet_loss(r, c, m, d, alpha=1.0):
#     """Compute magnet loss for batch.
    
#     Given a batch of features r consisting of m batches
#     each with d assigned examples and a cluster separation
#     gap of alpha, compute the total magnet loss and the per
#     example losses.
    
#     Args:
#         r: A batch of features.
#         c: Class labels for each example.
#         m: The number of clusters in the batch.
#         d: The number of examples in each cluster.
#         alpha: The cluster separation gap hyperparameter.
        
#     Returns:
#         total_loss: The total magnet loss for the batch.
#         losses: The loss for each example in the batch.
    
#     """
    
#     # Helper to compute indexes to select intra- and inter-cluster
#     # distances
#     def compute_comparison_inds():
#         same_cluster_inds = []
#         for i in range(m*d):
#             c = i / d
#             same_cluster_inds.append(c*d*m + c*d + (i % d))
#         diff_cluster_inds = sorted(set(range(m*m*d)) - set(same_cluster_inds))
        
#         return same_cluster_inds, diff_cluster_inds

        
#     # Take cluster means within the batch
#     cluster_means = tf.reduce_mean(tf.reshape(r, [m, d, -1]), 1)

#     # Compute squared distance of each example to each cluster centroid
#     sample_cluster_pair_inds = np.array(list(product(range(m*d), range(m))))
#     sample_costs = tf.squared_difference(
#         tf.gather(r, sample_cluster_pair_inds[:,1]),
#         tf.gather(cluster_means, sample_cluster_pair_inds[:,0]))
#     sample_costs = tf.reshape(tf.reduce_sum(sample_costs, 1), [m*d, m])

#     print sample_costs
#     return
    
#     # Compute intra- and inter-cluster comparison indexes
#     same_cluster_inds, diff_cluster_inds = compute_comparison_inds()

#     # Select distances of examples to their own centroid
#     intra_cluster_costs = tf.gather(sample_costs, same_cluster_inds)
#     intra_cluster_costs = tf.reshape(intra_cluster_costs, [m, d])
    
#     # Compute variance of intra-cluster squared distances
#     variance = tf.reduce_sum(intra_cluster_costs) / (m * d - 1)
#     var_normalizer = -1 / 2*variance**2

#     # Compute numerator of inner term
#     numerator = tf.exp(var_normalizer * intra_cluster_costs - alpha)

# #     # Select distances of examples to other centroids
# #     inter_cluster_costs = tf.reshape(tf.gather(sample_costs, diff_cluster_inds), [m, m-1, d])
    

#     # To compute denominator, select distances of examples to other class centroids
#     cluster_classes = tf.strided_slice(c, [0], [m*d], [d])
#     diff_class_mask = tf.logical_not(tf.equal(tf.expand_dims(c, 1), tf.expand_dims(cluster_classes, 0)))
#     denom_sample_costs = tf.exp(var_normalizer * tf.transpose(tf.reshape(sample_costs, [m, -1])))
#     diff_class_sample_costs = tf.select(diff_class_mask, denom_sample_costs, tf.zeros([m*d, m]))
#     denominator = tf.reduce_sum(diff_class_sample_costs, 1)

#     print numerator
#     print denominator
#     return numerator, denominator


# #     denominator = tf.reduce_sum(tf.exp(var_normalizer * inter_cluster_costs), 1)

#     # Compute example losses and total loss
#     losses = tf.nn.relu(-tf.log(numerator / denominator))
#     total_loss = tf.reduce_mean(losses)
    
#     return total_loss, losses
