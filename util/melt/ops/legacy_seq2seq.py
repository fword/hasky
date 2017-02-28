#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   seq2seq.py
#        \author   chenghuige  
#          \date   2016-12-20 11:24:54.024011
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest 

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: [batch_size, num_steps, num_decoder_symbols]. 
            if softmax_loss_function is not None then here is [batch_size, num_steps, emb_dim], actually is just outputs from dynamic_rnn 
    targets: [batch_size, num_steps]
    weights: [batch_size, num_steps]
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  """
  with ops.name_scope(name, "sequence_loss_by_example",
                      [logits, targets, weights]):
    if softmax_loss_function is None:
      #croosents [batch_size, num_steps]
      crossents = nn_ops.sparse_softmax_cross_entropy_with_logits(logits, targets)
    else:
      logits_shape = array_ops.shape(logits)
      batch_size = logits_shape[0]
      emb_dim = logits_shape[-1]
      #need reshape because unlike sparse_softmax_cross_entropy_with_logits, 
      #tf.nn.sampled_softmax_loss now only accept 2d [batch_size, dim] as logits input
      logits = array_ops.reshape(logits, [-1, emb_dim])
      targets = array_ops.reshape(targets, [-1, 1])
      #croosents [batch_size * num_steps]

      print('logits', logits)
      crossents = softmax_loss_function(logits, targets)
      # croosents [batch_size, num_steps]
      crossents = array_ops.reshape(crossents, [batch_size, -1])

    log_perps = math_ops.reduce_sum(math_ops.multiply(crossents, weights), 1)

    if average_across_timesteps:
      total_size = math_ops.reduce_sum(weights, 1)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: [batch_size, num_steps, num_decoder_symbols]. 
            if softmax_loss_function is not None then here is [batch_size, num_steps, emb_dim], actually is just outputs from dynamic_rnn 
    targets: [batch_size, num_steps]
    weights: [batch_size, num_steps]
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
  """
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets)[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost

#sample loss must be None 
def exact_predict_loss(self, logits, batch_size, num_steps):
  #logits = tf.reshape(logits, [batch_size, -1, self.vocab_size])
  i = tf.constant(0, dtype=tf.int32)
  condition = lambda i, log_probs: tf.less(i, num_steps)
  log_probs = tf.zeros([batch_size,], dtype=tf.float32)
  def body(i, log_probs):
    #-----below only ok in tf11.0 which can index by tensor TODO
    #step_logits = y[i, :, :]
    step_logits = tf.squeeze(tf.slice(logits, [0, i, 0], [-1, 1, -1]), [1])
    step_probs = tf.nn.softmax(step_logits)
    #step_targets = targets[i]
    step_targets = tf.squeeze(tf.slice(targets, [0, i], [-1, 1]), [1])
    selected_probs = melt.dynamic_gather2d(step_probs, step_targets)
    selected_log_probs = tf.log(tf.maximum(selected_probs, 1e-12))
    #step_mask = mask[:, i]
    #----FIXME
    step_mask = tf.squeeze(tf.slice(mask, [0, i], [-1, 1]), [1])
    log_probs += selected_log_probs * step_mask
    return tf.add(i, 1), tf.reshape(log_probs, [batch_size,])
  _, log_probs = tf.while_loop(condition, body, [i, log_probs])
  loss = -log_probs;
  return loss


import tensorflow as tf
#--for debug only now
def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state

def beam_decode(input, max_steps, initial_state, cell, loop_function, scope=None,
                beam_size=7, done_token=0, 
                output_projection=None,  length_normalization_factor=1.0,
                prob_as_score=True, topn=1):
    """
    Beam search decoder
    copy from https://gist.github.com/igormq/000add00702f09029ea4c30eba976e0a
    make small modifications, add more comments and add topn support, and 
    length_normalization_factor
    
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: This function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). 
        Signature -- loop_function(prev_symbol, i) = next
          * prev_symbol is a 1D Tensor of shape [batch_size*beam_size]
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size*beam_size, input_size].
      scope: Passed to seq2seq.rnn_decoder
      beam_size: An integer beam size to use for each example
      done_token: An integer token that specifies the STOP symbol
      
    Return:
      A tensor of dimensions [batch_size, len(decoder_inputs)] that corresponds to
      the 1-best beam for each batch.
      
    Known limitations:
      * The output sequence consisting of only a STOP symbol is not considered
        (zero-length sequences are not very useful, so this wasn't implemented)
      * The computation graph this creates is messy and not very well-optimized

    TODO: allow return top n result
    """
    decoder = BeamDecoder(input, max_steps, initial_state, beam_size=beam_size,
                          done_token=done_token, output_projection=output_projection,
                          length_normalization_factor=length_normalization_factor,
                          topn=topn)
    
    #_ = tf.nn.seq2seq.rnn_decoder(
    #_ = tf.contrib.legacy_seq2seq.rnn_decoder(
    _ = rnn_decoder(
        decoder.decoder_inputs,
        decoder.initial_state,
        cell=cell,
        loop_function = lambda prev, i: loop_function(decoder.take_step(prev, i), i),
        scope=scope
    )
    
    if topn == 1:
      score = decoder.logprobs_finished_beams
      path = decoder.finished_beams
    else:
      path, score = decoder.calc_topn()
    
    if prob_as_score:
      score = tf.exp(score)
    return path, score

#TODO:fully understand TensorArray usage and write dynamic BeamDecoder
class BeamDecoder():
  def __init__(self, input, max_steps, initial_state, beam_size=7, done_token=0,
              batch_size=None, num_classes=None, output_projection=None, 
              length_normalization_factor=1.0, topn=1):
    self.length_normalization_factor = length_normalization_factor
    self.topn = topn
    self.beam_size = beam_size
    self.batch_size = batch_size
    if self.batch_size is None:
        self.batch_size = tf.shape(input)[0]
    self.max_len = max_steps
    self.num_classes = num_classes
    self.done_token = done_token

    self.output_projection = output_projection
    
    self.past_logprobs = None
    self.past_symbols = None

    if topn == 1:
      self.finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32)
      self.logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')
    else:
      self.paths_list = []
      self.logprobs_list = []

    self.decoder_inputs = [None] * self.max_len
    #->[batch_size * beam_size, emb_dim]
    print('input in beamdecoder', input)
    self.decoder_inputs[0] = self.tile_along_beam(input)
    print('decoder_inputs[0]', self.decoder_inputs[0])

    assert len(initial_state) == 2, 'state is tuple of 2 elements'
    initial_state =  tf.concat(1, initial_state)
    self.initial_state = self.tile_along_beam(initial_state)

    try:
      self.initial_state = tf.split(self.initial_state, 2, 1)
    except Exception:
      self.initial_state = tf.split(1, 2, self.initial_state)
                      
  def tile_along_beam(self, tensor):
    """
    Helps tile tensors for each beam.
    for beam size 5
    x = tf.constant([[1.3, 2.4, 1.6], 
                    [3.2, 0.2, 1.5]])
    tile_along_beam(x).eval()
    array([
       [ 1.29999995,  2.4000001 ,  1.60000002],
       [ 1.29999995,  2.4000001 ,  1.60000002],
       [ 1.29999995,  2.4000001 ,  1.60000002],
       [ 1.29999995,  2.4000001 ,  1.60000002],
       [ 1.29999995,  2.4000001 ,  1.60000002],
       [ 3.20000005,  0.2       ,  1.5       ],
       [ 3.20000005,  0.2       ,  1.5       ],
       [ 3.20000005,  0.2       ,  1.5       ],
       [ 3.20000005,  0.2       ,  1.5       ],
       [ 3.20000005,  0.2       ,  1.5       ]
       ], dtype=float32)
    Args:
      tensor: a 2-D tensor, [batch_size x T]
    Return:
      An [batch_size*beam_size x T] tensor, where each row of the input
      tensor is copied beam_size times in a row in the output
    """
    res = tf.tile(tensor, [1, self.beam_size])
    res = tf.reshape(res, [-1, tf.shape(tensor)[1]])
    try:
      new_first_dim = tensor.get_shape()[0] * self.beam_size
    except Exception:
      new_first_dim = None
    res.set_shape((new_first_dim, tensor.get_shape()[1]))
    return res

  def calc_topn(self):
    #[batch_size, beam_size * (max_len-2)]
    logprobs = tf.concat(1, self.logprobs_list)

    #[batch_size, topn]
    top_logprobs, indices = tf.nn.top_k(logprobs, self.topn)


    length = self.beam_size * (self.max_len - 2)
    indice_offsets = tf.reshape(
      (tf.range(self.batch_size * length) // length) * length,
      [self.batch_size, length])

    indice_offsets = indice_offsets[:,0:self.topn]

    #[batch_size, max_len(length index) - 2, beam_size, max_len]
    paths = tf.concat(1, self.paths_list)
    #top_paths = paths[:, 0:self.topn, 2, :]
    
    #paths = tf.reshape(paths, [-1, (self.max_len  - 2) * self.beam_size, self.max_len])
    #paths = tf.reshape(paths, [self.batch_size, -1, self.max_len])

    paths = tf.reshape(paths, [-1, self.max_len])

    top_paths = tf.gather(paths, indice_offsets + indices)

    return top_paths, top_logprobs
        
  def take_step(self, prev, i):
    output_projection = self.output_projection
    if output_projection is not None:
      #[batch_size * beam_size, num_units] -> [batch_size * beam_size, num_classes]
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])

    #[batch_size * beam_size, num_classes], here use log sofmax
    logprobs = tf.nn.log_softmax(prev)
    
    if self.num_classes is None:
      self.num_classes = tf.shape(logprobs)[1]

    #->[batch_size, beam_size, num_classes]
    logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.num_classes])
    logprobs_batched.set_shape((None, self.beam_size, None))
    
    # Note: masking out entries to -inf plays poorly with top_k, so just subtract out a large number.
    nondone_mask = tf.reshape(
        tf.cast(tf.equal(tf.range(self.num_classes), self.done_token), tf.float32) * -1e18,
        [1, 1, self.num_classes])
    if self.past_logprobs is not None:
      #logprobs_batched [batch_size, beam_size, num_classes] -> [batch_size, beam_size, num_classes]  
      #past_logprobs    [batch_size, beam_size] -> [batch_size, beam_size, 1]
      logprobs_batched = logprobs_batched + tf.expand_dims(self.past_logprobs, 2)

      self.past_logprobs, indices = tf.nn.top_k(
          tf.reshape(logprobs_batched + nondone_mask, [-1, self.beam_size * self.num_classes]),
          self.beam_size)       
    else:
      #[batch_size, beam_size, num_classes] -> [batch_size, num_classes]
      #-> past_logprobs[batch_size, beam_size], indices[batch_size, beam_size]
      self.past_logprobs, indices = tf.nn.top_k(
          (logprobs_batched + nondone_mask)[:,0,:],
          self.beam_size)

    # For continuing to the next symbols [batch_size, beam_size]
    symbols = indices % self.num_classes
    #from wich beam it comes  [batch_size, beam_size]
    parent_refs = indices // self.num_classes
    
    if self.past_symbols is not None:
      #/notebooks/mine/ipynotebook/tensorflow/beam-search2.ipynb below for mergeing path
      #here when i >= 2
      # tf.reshape(
      #           (tf.range(3 * 5) // 5) * 5,
      #           [3, 5]
      #       ).eval()
      # array([[ 0,  0,  0,  0,  0],
      #        [ 5,  5,  5,  5,  5],
      #        [10, 10, 10, 10, 10]], dtype=int32)
      parent_refs_offsets = tf.reshape(
          (tf.range(self.batch_size * self.beam_size) // self.beam_size) * self.beam_size,
          [self.batch_size, self.beam_size])
      
      #self.past_symbols [batch_size, beam_size, 1]
      past_symbols_batch_major = tf.reshape(self.past_symbols, [-1, i-1])
      beam_past_symbols = tf.gather(past_symbols_batch_major,
                                parent_refs + parent_refs_offsets)

      if self.topn > 1:
        #[batch_size, beam_size, max_len]
        path = tf.concat(2, [self.past_symbols, 
                             tf.tile(tf.ones_like(tf.expand_dims(symbols, 2))* self.done_token, 
                             [1, 1, self.max_len - i + 1])])

        #print(i, 'abc', path.get_shape())
        #[batch_size, 1, beam_size, max_len]
        path = tf.expand_dims(path, 1)
        self.paths_list.append(path)

      #[batch_size, bam_size, i] the best beam_size paths until step i
      self.past_symbols = tf.concat(2, [beam_past_symbols, tf.expand_dims(symbols, 2)])


      # For finishing the beam 
      #[batch_size, beam_size]
      logprobs_done = logprobs_batched[:,:,self.done_token]
      if self.topn > 1:
        self.logprobs_list.append(logprobs_done / i ** self.length_normalization_factor)
      else:
        done_parent_refs = tf.cast(tf.argmax(logprobs_done, 1), tf.int32)
        done_parent_refs_offsets = tf.range(self.batch_size) * self.beam_size

        done_past_symbols = tf.gather(past_symbols_batch_major,
                                      done_parent_refs + done_parent_refs_offsets)

        #[batch_size, max_len]
        symbols_done = tf.concat(1, [done_past_symbols,
                                     tf.ones_like(done_past_symbols[:,0:1]) * self.done_token,
                                     tf.tile(tf.zeros_like(done_past_symbols[:,0:1]),
                                             [1, self.max_len - i])
                                    ])

        #print(i, 'cde', symbols_done.get_shape())

        #[batch_size, beam_size] -> [batch_size,]
        logprobs_done_max = tf.reduce_max(logprobs_done, 1)
      
        if self.length_normalization_factor > 0:
          logprobs_done_max /= i ** self.length_normalization_factor

        self.finished_beams = tf.select(logprobs_done_max > self.logprobs_finished_beams,
                                        symbols_done,
                                        self.finished_beams)
        self.logprobs_finished_beams = tf.maximum(logprobs_done_max, self.logprobs_finished_beams)
    else:
      #here when i == 1
      #[batch_size, beam_size] -> [batch_size, beam_size, 1]
      self.past_symbols = tf.expand_dims(symbols, 2)
      # NOTE: outputing a zero-length sequence is not supported for simplicity reasons

    #->[batch_size * beam_size,]
    symbols_flat = tf.reshape(symbols, [-1])

    return symbols_flat