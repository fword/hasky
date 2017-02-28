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
#since not from melt.layers.layers import * this is safe
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import  normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

#@TODO add scope
# def fully_connected(x, output_size, activation=tf.nn.relu, scope=None):
#   #@TODO -1 or last dim ? NotImplementedError("Negative indices are currently unsupported")
#   #input_dim = tf.shape(x)[-1]
#   #@TODO how is slim.fully_connected get inputdim and use..
#   #below will now work int() argument must be a string or a number, not 'Tensor' [input_dim, output_size])
#   #input_dim = tf.shape(x)[1]
#   #check contrib\layers\python\layers\layers.py
#   scope = 'fc' if scope is None else scope
#   with tf.variable_scope(scope):
#     input_dim = utils.last_dimension(x.get_shape(), min_rank=2)
#     if isinstance(x, tf.Tensor):
#       w_h = melt.get_weights('w_h', [input_dim, output_size])
#     else:
#       with tf.device('/cpu:0'):
#         w_h = melt.get_weights('w_h', [input_dim, output_size]) 
#     b_h = melt.get_bias('b_h', [output_size])
#     return activation(melt.matmul(x, w_h) + b_h)

def fully_connected(inputs,
                    num_outputs,
                    input_dim=None,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):

  use_bias = biases_initializer is not None

  #--------TODO: use commented code as layers.fully_connected then, for app code you must manully pass scope like 'mlp' otherwise evaluate will fail try to use Mlp_1 not resue=True b
  #--------by tf.variable_scope.reuse_variables() see http://stackoverflow.com/questions/40536665/tensorflow-varscope-reuse-variables
  #with variable_scope.variable_scope(
  #  scope, 'Mlp', [inputs],
  #  reuse=reuse) as vs:
  scope = 'fully_connected' if scope is None else scope
  with tf.variable_scope(scope):
    is_dense_input = True if isinstance(inputs, tf.Tensor) else False
    dtype=inputs.dtype.base_dtype if is_dense_input else inputs[1].values.dtype.base_dtype
    #sparse input must tell input_dim
    assert is_dense_input or input_dim is not None 
    if is_dense_input:
      shape = inputs.get_shape().as_list() 
      input_dim = shape[-1].value
      assert len(shape) == 2, "now only consider X shape dim as 2, TODO: make > 2 ok like layers.fully_connected"

    #-----------deal first hidden
    if is_dense_input:
     w_h =  tf.get_variable('weight_hidden',
                            shape=[input_dim, num_outputs],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            dtype=dtype,
                            trainable=trainable)
    else:
     with tf.device('/cpu:0'):
       w_h =  tf.get_variable('weight_hidden',
                              shape=[input_dim, num_outputs],
                              initializer=weights_initializer,
                              regularizer=weights_regularizer,
                              dtype=dtype,
                              trainable=trainable)

    if use_bias:
     b_h = tf.get_variable('bias_hidden',
                           shape=[num_outputs,],
                           initializer=biases_initializer,
                           regularizer=biases_regularizer,
                           dtype=dtype,
                           trainable=trainable)

    outputs = melt.matmul(inputs, w_h)
    if use_bias:
     outputs = nn.bias_add(outputs, b_h)
    if activation_fn is not None:
     outputs = activation_fn(outputs)  # pylint: disable=not-callable

    return outputs

linear = functools.partial(fully_connected, activation_fn=None)

def mlp(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp' if scope is None else scope
  with tf.variable_scope(scope):
    hidden = fully_connected(x, hidden_size, activation)
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    b_o = melt.get_bias('b_o', [output_size])
    return tf.nn.xw_plus_b(hidden, w_o, b_o)

def mlp_nobias(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp_nobias' if scope is None else scope
  with tf.variable_scope(scope):
    input_dim = utils.last_dimension(x.get_shape(), min_rank=2)
    if isinstance(x, tf.Tensor):
      w_h = melt.get_weights('w_h', [input_dim, hidden_size])
    else:
      with tf.device('/cpu:0'):
        w_h = melt.get_weights('w_h', [input_dim, hidden_size]) 
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    return  melt.mlp_forward_nobias(x, w_h, w_o, activation)