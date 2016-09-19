import nose
from nose.tools import *
from magnet_ops import *


def test_magnet_loss():
    """Test magnet loss ops."""    
    rand = np.random.RandomState(42)

    # Hyperparams
    m = 6
    d = 4
    K = 5

    # Sample test data
    rdata = rand.random_sample([m*d, 8])
    clusters = np.repeat(range(m), d)

    cluster_classes1 = range(m)
    classes1 = np.repeat(cluster_classes1, d)

    cluster_classes2 = [0, 1, 1, 3, 4, 5]
    classes2 = np.repeat(cluster_classes2, d)

    # Placeholders and ops
    p_r = tf.placeholder(tf.float32, [m*d, 8])
    p_classes = tf.placeholder(tf.int32, [m*d])
    p_clusters = tf.placeholder(tf.int32, [m*d])
    p_cluster_classes = tf.placeholder(tf.int32, [m])
    p_alpha = tf.placeholder(tf.float32, [])

    total_loss1, example_losses1 = minibatch_magnet_loss(p_r, p_classes, m, d, p_alpha)
    total_loss2, example_losses2 = magnet_loss(p_r, p_classes, p_clusters,
                                               p_cluster_classes, m, p_alpha)


    sess = tf.InteractiveSession()

    # Simple case
    feed_dict = {p_r: rdata, p_classes: classes1, p_clusters: clusters,
                 p_cluster_classes: cluster_classes1, p_alpha: 1.0}
    loss1, loss2 = sess.run([total_loss1, total_loss2], feed_dict=feed_dict)
    assert_almost_equal(loss1, 2.1448281)
    assert_almost_equal(loss2, 2.1448281)

    # Test alpha
    feed_dict = {p_r: rdata, p_classes: classes1, p_clusters: clusters,
                 p_cluster_classes: cluster_classes1, p_alpha: 10.0}
    loss1, loss2 = sess.run([total_loss1, total_loss2], feed_dict=feed_dict)
    assert_almost_equal(loss1, 11.144104)
    assert_almost_equal(loss2, 11.144104)

    # Test multiple clusters per class
    feed_dict = {p_r: rdata, p_classes: classes2, p_clusters: clusters,
                 p_cluster_classes: cluster_classes2, p_alpha: 1.0}
    loss1, loss2 = sess.run([total_loss1, total_loss2], feed_dict=feed_dict)
    assert_almost_equal(loss1, 2.0619006)
    assert_almost_equal(loss2, 2.0619006)

    sess.close()
    tf.reset_default_graph()

if __name__ == '__main__':
    nose.main()
