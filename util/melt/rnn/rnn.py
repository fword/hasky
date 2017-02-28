#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2016-12-23 14:02:57.513674
#   \Description  
# ==============================================================================

"""
rnn encoding
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from melt.ops import dynamic_last_relevant

import copy
  
class EncodeMethod:
  forward = 0
  backward = 1
  bidrectional = 2 
  bidrectional_sum = 3

class OutputMethod:
  sum = 0
  last = 1
  first = 2
  all = 3

def forward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    inputs, 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)
  
  #--seems slower convergence and not good result when only using last output, so change to use sum
  if output_method == OutputMethod.sum:
    return tf.reduce_sum(outputs, 1), state
  elif output_method == OutputMethod.last:
    return dynamic_last_relevant(outputs, sequence_length), state
  elif output_method == OutputMethod.first:
    return outputs[:, 0, :], state
  else:
    return outputs, state

def backward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    tf.reverse_sequence(inputs, sequence_length, 1), 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)

  #--seems slower convergence and not good result when only using last output, so change to use sum
  if output_method == OutputMethod.sum:
    return tf.reduce_sum(outputs, 1), state
  elif output_method == OutputMethod.last:
    return dynamic_last_relevant(outputs, sequence_length), state
  elif output_method == OutputMethod.first:
    return  outputs[:, 0, :], state
  else:
    return outputs, state

def bidrectional_encode(cell_fw, 
                        cell_bw, 
                        inputs, 
                        sequence_length, 
                        initial_state_fw=None, 
                        initial_state_bw=None, 
                        dtype=None,
                        output_method=OutputMethod.last,
                        use_sum=False):
  if cell_bw is None:
    cell_bw = copy.deepcopy(cell_fw)
  if initial_state_bw is None:
    initial_state_bw = initial_state_fw

  outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell_fw,
    cell_bw=cell_bw,
    inputs=inputs,
    initial_state_fw=initial_state_fw,
    initial_state_bw=initial_state_bw,
    dtype=dtype,
    sequence_length=sequence_length)

  output_fws, output_bws = outputs

  if output_method == OutputMethod.sum:
    output_forward = tf.reduce_sum(output_fws, 1) 
  elif output_method == OutputMethod.last:
    output_forward = dynamic_last_relevant(output_fws, sequence_length)
  elif output_method == OutputMethod.first:
    output_forward = output_fws[:, 0, :]
  else:
    output_forward = output_fws

  if output_method == OutputMethod.sum:
    output_backward = tf.reduce_sum(output_bws, 1) 
  elif output_method == OutputMethod.last:
    output_backward = dynamic_last_relevant(output_bws, sequence_length)
  elif output_method == OutputMethod.first:
    output_backward = output_bws[:, 0, :]
  else:
    output_backward = output_bws

  if use_sum:
    output = output_forward + output_backward
  else:
    output = tf.concat(-1, [output_forward, output_backward])

  return output, states[0]

def encode(cell, 
           inputs, 
           sequence_length, 
           initial_state=None, 
           cell_bw=None, 
           inital_state_bw=None, 
           dtype=None,
           encode_method=EncodeMethod.forward, 
           output_method=OutputMethod.last):
    
    #needed for bidirectional_dynamic_rnn and backward method
    #without it Input 'seq_lengths' of 'ReverseSequence' Op has type int32 that does not match expected type of int64.
    #int tf.reverse_sequence seq_lengths: A `Tensor` of type `int64`.
    if initial_state is None and dtype is None:
      dtype = tf.float32
    sequence_length = tf.cast(sequence_length, tf.int64)
    if encode_method == EncodeMethod.forward:
      return forward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.backward:
      return backward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.bidrectional:
      return bidrectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method)
    elif encode_method == EncodeMethod.bidrectional_sum:
      return bidrectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method,
                                 use_sum=True)
    else:
      raise ValueError('Unsupported rnn encode method:', encode_method)