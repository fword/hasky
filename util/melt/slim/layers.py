#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   layers.py
#        \author   chenghuige  
#          \date   2016-08-19 23:22:44.032101
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
import tensorflow.contrib.slim as slim

def mlp(x, 
        dims, 
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer,
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True, 
        scope=None):
  scope = 'mlp' if scope is None else scope
  with tf.variable_scope(scope):
    for i in xrange(len(dims) -1):
      x = slim.fully_connected(x, dims[i], 
                               activation_fn=activation_fn, 
                               weights_initializer=weights_initializer,
                               biases_initializer=biases_initializer, 
                               scope='fc_%d'%i)
    return slim.linear(x, 
                       dims[-1], 
                       weights_initializer=weights_initializer,
                       biases_initializer=biases_initializer,
                       scope='linear')
