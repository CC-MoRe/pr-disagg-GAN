

import numpy as np
import tensorflow as tf


def per_gridpoint_softmax(x):

    # x has dimension (batch_size,tres,ngridpoints) pr (batch_size,tres,ngridpoints)
    exps = tf.exp(x)
    # now normalize over tres dimension
    norm = tf.reduce_sum(exps,1)  # this now has shape (batch_size,lat,lon)
    # to align the shapes of norm and exps, we need to swap dimensions of exps, so that tres is first
    # simplest is to wimply swap the first 2 dimensions
    # tf.transpose needs the full permutation vectors to make this work for general ranks of x
    perm = tf.concat([[1,0], tf.range(2, tf.rank(x))], 0)
    exps = tf.transpose(exps, perm)
    out = exps / norm
    out = tf.transpose(out, perm)
    return out


x = tf.random.normal((4,3,5,6))
z = per_gridpoint_softmax(x)
z = np.array(z)
assert(z.shape==x.shape)
assert(z.sum(1))
np.testing.assert_allclose(z.sum(1),1, rtol=1e-5)