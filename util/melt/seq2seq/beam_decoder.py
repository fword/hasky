#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   beam_decoder.py
#        \author   chenghuige  
#          \date   2017-01-16 14:15:32.474973
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import array_ops

from melt.seq2seq import beam_decoder_fn_inference

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
                prob_as_score=True, topn=1, 
                attention_construct_fn=None, attention_keys=None, attention_values=None):
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
                          done_token=done_token, 
                          output_projection=output_projection,
                          length_normalization_factor=length_normalization_factor,
                          topn=topn, 
                          attention_construct_fn=attention_construct_fn,
                          attention_keys=attention_keys, 
                          attention_values=attention_values)
    
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

import tensorflow as tf
def dynamic_beam_decode(input, max_steps, initial_state, cell, embedding, scope=None,
                beam_size=7, done_token=0, num_classes=None,
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

    #TODO may be not input num_classes?
    assert num_classes is not None

    decoder = BeamDecoder(input, max_steps, initial_state, 
                          done_token=done_token, num_classes=num_classes,
                          output_projection=output_projection,
                          length_normalization_factor=length_normalization_factor,
                          topn=topn)
    
    def output_fn(output, i):
      next_input_id = decoder.take_step(output, i)
      return decoder.output, next_input_id

    decoder_fn_inference = beam_decoder_fn_inference(
                  output_fn=output_fn,
                  first_input=decoder.decoder_inputs[0],
                  encoder_state=decoder.initial_state,
                  embeddings=embedding,
                  end_of_sequence_id=done_token,
                  maximum_length=max_steps,
                  num_decoder_symbols=num_classes,
                  decoder=decoder,
                  dtype=tf.int32)

    (decoder_outputs_inference, decoder_state_inference,
      decoder_context_state_inference) = (tf.contrib.seq2seq.dynamic_rnn_decoder(
               cell=cell,
               decoder_fn=decoder_fn_inference,
               scope=scope))

    score = decoder.logprobs_finished_beams
    path = decoder_context_state_inference

    if prob_as_score:
      score = tf.exp(score)
    return path, score

def _init_attention(encoder_state):
  """Initialize attention. Handling both LSTM and GRU.

  Args:
    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.

  Returns:
    attn: initial zero attention vector.
  """

  # Multi- vs single-layer
  # TODO(thangluong): is this the best way to check?
  if isinstance(encoder_state, tuple):
    top_state = encoder_state[-1]
  else:
    top_state = encoder_state

  # LSTM vs GRU
  if isinstance(top_state, core_rnn_cell_impl.LSTMStateTuple):
    attn = array_ops.zeros_like(top_state.h)
  else:
    attn = array_ops.zeros_like(top_state)

  return attn

class BeamDecoder():
  def __init__(self, input, max_steps, initial_state, beam_size=7, done_token=0,
              batch_size=None, num_classes=None, output_projection=None, 
              length_normalization_factor=1.0, topn=1,
              attention_construct_fn=None, attention_keys=None, attention_values=None):
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
      self.finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32) + 837
      self.logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')
    else:
      self.paths_list = []
      self.logprobs_list = []

    self.decoder_inputs = [None] * self.max_len
    #->[batch_size * beam_size, emb_dim]
    if attention_construct_fn is not None:
      input = tf.concat([input, _init_attention(initial_state)], 1)
    self.decoder_inputs[0] = self.tile_along_beam(input)

    assert len(initial_state) == 2, 'state is tuple of 2 elements'
    initial_state =  tf.concat(initial_state, 1)
    self.initial_state = self.tile_along_beam(initial_state)

    self.initial_state = tf.split(self.initial_state, 2, 1)
    self.initial_state = tuple(self.initial_state)

    self.attention_construct_fn = attention_construct_fn
    self.attention_keys = self.tile_along_beam_attention(attention_keys) if attention_keys is not None else None
    self.attention_values = self.tile_along_beam_attention(attention_values) if attention_values is not None else None
                      
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
    #TODO: why below not work
    #res = tf.reshape(res, [-1, tensor.get_shape()[1]]) 
    try:
      new_first_dim = tensor.get_shape()[0] * self.beam_size
    except Exception:
      new_first_dim = None
    res.set_shape((new_first_dim, tensor.get_shape()[1]))
    return res

  #TODO: merge with title_along_beam?
  def tile_along_beam_attention(self, tensor):
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
    res = tf.tile(tensor, [1, self.beam_size, 1])
    tensor_shape = tf.shape(tensor)
    res = tf.reshape(res, [-1, tensor_shape[1], tensor_shape[2]])
    #TODO: why below not work
    #res = tf.reshape(res, [-1, tensor.get_shape()[1]]) 
    try:
      new_first_dim = tensor.get_shape()[0] * self.beam_size
    except Exception:
      new_first_dim = None
    res.set_shape((new_first_dim, tensor.get_shape()[1], tensor.get_shape()[2]))
    return res

  def calc_topn(self):
    #[batch_size, beam_size * (max_len-2)]
    logprobs = tf.concat(self.logprobs_list, 1)

    #[batch_size, topn]
    top_logprobs, indices = tf.nn.top_k(logprobs, self.topn)


    length = self.beam_size * (self.max_len - 2)
    indice_offsets = tf.reshape(
      (tf.range(self.batch_size * length) // length) * length,
      [self.batch_size, length])

    indice_offsets = indice_offsets[:,0:self.topn]

    #[batch_size, max_len(length index) - 2, beam_size, max_len]
    paths = tf.concat(self.paths_list, 1)
    #top_paths = paths[:, 0:self.topn, 2, :]
    
    #paths = tf.reshape(paths, [-1, (self.max_len  - 2) * self.beam_size, self.max_len])
    #paths = tf.reshape(paths, [self.batch_size, -1, self.max_len])

    paths = tf.reshape(paths, [-1, self.max_len])

    top_paths = tf.gather(paths, indice_offsets + indices)

    return top_paths, top_logprobs
        
  def take_step(self, prev, i):
    output_projection = self.output_projection
    if self.attention_construct_fn is not None:
      #prev is as cell_output, see contrib\seq2seq\python\ops\attention_decoder_fn.py
      attention = self.attention_construct_fn(prev, self.attention_keys, self.attention_values)
      prev = attention
    else:
      attention = None
    if output_projection is not None:
      #[batch_size * beam_size, num_units] -> [batch_size * beam_size, num_classes]
      output = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
    else:
      output = prev

    self.output = output

    #[batch_size * beam_size, num_classes], here use log sofmax
    logprobs = tf.nn.log_softmax(output)
    
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
        path = tf.concat([self.past_symbols, 
                             tf.tile(tf.ones_like(tf.expand_dims(symbols, 2))* self.done_token, 
                             [1, 1, self.max_len - i + 1])], 2)

        #print(i, 'abc', path.get_shape())
        #[batch_size, 1, beam_size, max_len]
        path = tf.expand_dims(path, 1)
        self.paths_list.append(path)

      #[batch_size, bam_size, i] the best beam_size paths until step i
      self.past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2)], 2)


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
        symbols_done = tf.concat([done_past_symbols,
                                     tf.ones_like(done_past_symbols[:,0:1]) * self.done_token,
                                     tf.tile(tf.zeros_like(done_past_symbols[:,0:1]),
                                             [1, self.max_len - i])
                                    ], 1)

        #print(i, 'cde', symbols_done.get_shape())

        #[batch_size, beam_size] -> [batch_size,]
        logprobs_done_max = tf.reduce_max(logprobs_done, 1)
      
        if self.length_normalization_factor > 0:
          logprobs_done_max /= i ** self.length_normalization_factor

        #[batch_size, max_len]
        self.finished_beams = tf.where(logprobs_done_max > self.logprobs_finished_beams,
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

    if attention is None:
      return symbols_flat
    else:
      return symbols_flat, attention
  

