#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dynamic_beam_decoder.py
#        \author   chenghuige  
#          \date   2017-01-25 06:44:17.350088
#   \Description  
# ==============================================================================

  
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import tensorflow as tf


# import abc

# import six

# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import tensor_array_ops
# from tensorflow.python.util import nest

# __all__ = ["Decoder", "dynamic_decode_rnn"]


# def _transpose_batch_time(x):
#   """Transpose the batch and time dimensions of a Tensor.

#   Retains as much of the static shape information as possible.

#   Args:
#     x: A tensor of rank 2 or higher.

#   Returns:
#     x transposed along the first two dimensions.

#   Raises:
#     ValueError: if `x` is rank 1 or lower.
#   """
#   x_static_shape = x.get_shape()
#   if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
#     raise ValueError(
#         "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
#         (x, x_static_shape))
#   x_rank = array_ops.rank(x)
#   x_t = array_ops.transpose(
#       x, array_ops.concat_v2(
#           ([1, 0], math_ops.range(2, x_rank)), axis=0))
#   x_t.set_shape(
#       tensor_shape.TensorShape([
#           x_static_shape[1].value, x_static_shape[0].value
#       ]).concatenate(x_static_shape[2:]))
#   return x_t


# @six.add_metaclass(abc.ABCMeta)
# class Decoder(object):
#   """An RNN Decoder abstract interface object."""

#   @property
#   def batch_size(self):
#     """The batch size of the inputs returned by `sample`."""
#     raise NotImplementedError

#   @property
#   def output_size(self):
#     """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
#     raise NotImplementedError

#   @property
#   def output_dtype(self):
#     """A (possibly nested tuple of...) dtype[s]."""
#     raise NotImplementedError

#   @abc.abstractmethod
#   def initialize(self, name=None):
#     """Called before any decoding iterations.

#     Args:
#       name: Name scope for any created operations.

#     Returns:
#       `(finished, first_inputs, initial_state)`.
#     """
#     raise NotImplementedError

#   @abc.abstractmethod
#   def step(self, time, inputs, state):
#     """Called per step of decoding (but only once for dynamic decoding).

#     Args:
#       time: Scalar `int32` tensor.
#       inputs: Input (possibly nested tuple of) tensor[s] for this time step.
#       state: State (possibly nested tuple of) tensor[s] from previous time step.

#     Returns:
#       `(outputs, next_state, next_inputs, finished)`.
#     """
#     raise NotImplementedError


# def _create_zero_outputs(size, dtype, batch_size):
#   """Create a zero outputs Tensor structure."""
#   def _t(s):
#     return (s if isinstance(s, ops.Tensor) else constant_op.constant(
#         tensor_shape.TensorShape(s).as_list(),
#         dtype=dtypes.int32,
#         name="zero_suffix_shape"))

#   def _create(s, d):
#     return array_ops.zeros(
#         array_ops.concat(
#             ([batch_size], _t(s)), axis=0), dtype=d)

#   return nest.map_structure(_create, size, dtype)


# def dynamic_decode_rnn_beam_search(decoder,
#                        output_time_major=False,
#                        parallel_iterations=32,
#                        swap_memory=False):
#   """Perform dynamic decoding with `decoder`.

#   Args:
#     decoder: A `Decoder` instance.
#     output_time_major: Python boolean.  Default: `False` (batch major).  If
#       `True`, outputs are returned as time major tensors (this mode is faster).
#       Otherwise, outputs are returned as batch major tensors (this adds extra
#       time to the computation).
#     parallel_iterations: Argument passed to `tf.while_loop`.
#     swap_memory: Argument passed to `tf.while_loop`.

#   Returns:
#     `(final_outputs, final_state)`.

#   Raises:
#     TypeError: if `decoder` is not an instance of `Decoder`.
#   """
#   if not isinstance(decoder, Decoder):
#     raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
#                     type(decoder))

#   zero_outputs = _create_zero_outputs(decoder.output_size, decoder.output_dtype,
#                                       decoder.batch_size)

#   initial_finished, initial_inputs, initial_past_symbols, \
#    initial_past_logprobs, initial_finished_beams, initial_logprobs_finished_beams, initial_state= decoder.initialize()
#   initial_time = constant_op.constant(0, dtype=dtypes.int32)

#   def _shape(batch_size, from_shape):
#     if not isinstance(from_shape, tensor_shape.TensorShape):
#       return tensor_shape.TensorShape(None)
#     else:
#       batch_size = tensor_util.constant_value(
#           ops.convert_to_tensor(
#               batch_size, name="batch_size"))
#       return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

#   def _create_ta(s, d):
#     return tensor_array_ops.TensorArray(
#         dtype=d, size=0, dynamic_size=True,
#         element_shape=_shape(decoder.batch_size, s))

#   initial_outputs_ta = nest.map_structure(
#       _create_ta, decoder.output_size, decoder.output_dtype)

#   def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
#                 finished, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
#     return math_ops.logical_not(math_ops.reduce_all(finished))

#   def body(time, outputs_ta, state, inputs, finished, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
#     """Internal while_loop body.

#     Args:
#       time: scalar int32 tensor.
#       outputs_ta: structure of TensorArray.
#       state: (structure of) state tensors and TensorArrays.
#       inputs: (structure of) input tensors.
#       finished: 1-D bool tensor.

#     Returns:
#       `(time + 1, outputs_ta, next_state, next_inputs, next_finished)`.
#     """
#     (next_outputs, decoder_state, next_inputs, decoder_finished, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams) = decoder.step(
#         time, inputs, state, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)
#     next_finished = math_ops.logical_or(decoder_finished, finished)

#     nest.assert_same_structure(state, decoder_state)
#     nest.assert_same_structure(outputs_ta, next_outputs)
#     nest.assert_same_structure(inputs, next_inputs)

#     # Zero out output values past finish
#     emit = nest.map_structure(
#         lambda out, zero: array_ops.where(finished, zero, out), next_outputs,
#         zero_outputs)

#     # Copy through states past finish
#     def _maybe_copy_state(new, cur):
#       return (new if isinstance(cur, tensor_array_ops.TensorArray) else
#               array_ops.where(finished, cur, new))

#     next_state = nest.map_structure(_maybe_copy_state, decoder_state, state)
#     outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
#                                     outputs_ta, emit)
#     return (time + 1, outputs_ta, next_state, next_inputs, next_finished, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)

#   res = control_flow_ops.while_loop(
#       condition,
#       body,
#       loop_vars=[
#           initial_time, initial_outputs_ta, initial_state, initial_inputs,
#           initial_finished, initial_past_symbols, initial_past_logprobs, 
#           initial_finished_beams, initial_logprobs_finished_beams
#       ],
#       parallel_iterations=parallel_iterations,
#       swap_memory=swap_memory)

#   final_outputs_ta = res[1]
#   final_state = res[2]

#   final_ids = res[-2]
#   final_score = res[-1]

#   final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
#   if not output_time_major:
#     final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

#   return final_outputs, final_state, final_ids, final_score


# import abc
# import collections

# import six

# from tensorflow.contrib.rnn import core_rnn_cell
# from tensorflow.contrib.seq2seq.python.ops import decoder
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import tensor_array_ops
# from tensorflow.python.util import nest
# from tensorflow.python.ops import variable_scope

# _transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access

# @six.add_metaclass(abc.ABCMeta)
# class Sampler(object):

#   @property
#   def batch_size(self):
#     pass

#   @abc.abstractmethod
#   def initialize(self):
#     pass

#   @abc.abstractmethod
#   def sample(self, time, outputs, state):
#     pass


# class BasicSamplingDecoder(Decoder):
#   """Basic sampling decoder."""

#   def __init__(self, cell, sampler, initial_state, scope=None):
#     """Initialize BasicSamplingDecoder.

#     Args:
#       cell: An `RNNCell` instance.
#       sampler: A `Sampler` instance.
#       initial_state: A (possibly nested tuple of...) tensors and TensorArrays.

#     Raises:
#       TypeError: if `cell` is not an instance of `RNNCell` or `sampler`
#         is not an instance of `Sampler`.
#     """
#     if not isinstance(cell, core_rnn_cell.RNNCell):
#       raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
#     if not isinstance(sampler, Sampler):
#       raise TypeError("sampler must be a Sampler, received: %s" %
#                       type(sampler))
#     self._cell = cell
#     self._sampler = sampler
#     self._initial_state = initial_state
#     self._scope = scope

#   @property
#   def batch_size(self):
#     return self._sampler.batch_size

#   @property
#   def output_size(self):
#     # Return the cell output and the id
#     return self._cell.output_size

#   @property
#   def output_dtype(self):
#     # Assume the dtype of the cell is the output_size structure
#     # containing the input_state's first component's dtype.
#     # Return that structure and int32 (the id)
#     dtype = nest.flatten(self._initial_state)[0].dtype
#     return dtype

#   def initialize(self, name=None):
#     return self._sampler.initialize() + (self._initial_state,)

#   def step(self, time, inputs, state, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
#     """Perform a decoding step.

#     Args:
#       time: scalar `int32` tensor.
#       inputs: A (structure of) input tensors.
#       state: A (structure of) state tensors and TensorArrays.

#     Returns:
#       `(outputs, next_state, next_inputs, finished)`.
#     """
#     with variable_scope.variable_scope(self._scope or "rnn_decoder"):
#       outputs, next_state = self._cell(inputs, state)
#       next_time = time + 1
#       (finished, next_inputs, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams) = self._sampler.sample(
#           next_time, outputs, state, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)
#       return (outputs, next_state, next_inputs, finished, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)

# def _shape(batch_size, from_shape):
#   if not isinstance(from_shape, tensor_shape.TensorShape):
#     return tensor_shape.TensorShape(None)
#   else:
#     batch_size = tensor_util.constant_value(
#         ops.convert_to_tensor(
#             batch_size, name="batch_size"))
#     return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

# class DynamicBeamSampler(Sampler):
#   def __init__(self, embeddings, input, max_steps, initial_state, beam_size=7, done_token=0,
#               batch_size=None, num_classes=None, output_projection=None, 
#               length_normalization_factor=1.0, topn=1):
#     self.embeddings = embeddings
#     self.length_normalization_factor = length_normalization_factor
#     self.topn = topn
#     self.beam_size = beam_size
#     self._batch_size = batch_size
#     if self._batch_size is None:
#       self._batch_size = tf.shape(input)[0]
#     self.max_len = max_steps
#     self.num_classes = num_classes
#     self.done_token = done_token

#     self.output_projection = output_projection
    
#     self.past_logprobs = tf.zeros([self.batch_size, beam_size], tf.float32)
#     self.past_symbols = tensor_array_ops.TensorArray(
#         dtype=tf.int32, size=0, dynamic_size=True,
#         element_shape=_shape(self.batch_size, self.beam_size))

#     self.finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32) + 837
#     self.logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')

#     self.decoder_inputs = [None] * self.max_len
#     #->[batch_size * beam_size, emb_dim]
#     self.decoder_inputs[0] = self.tile_along_beam(input)

#     assert len(initial_state) == 2, 'state is tuple of 2 elements'
#     initial_state =  tf.concat(initial_state, 1)
#     self.initial_state = self.tile_along_beam(initial_state)

#     self.initial_state = tf.split(self.initial_state, 2, 1)
#     self.initial_state = tuple(self.initial_state)

#     self.output = None

#   @property
#   def batch_size(self):
#     return self._batch_size
                      
#   def tile_along_beam(self, tensor):
#     """
#     Helps tile tensors for each beam.
#     for beam size 5
#     x = tf.constant([[1.3, 2.4, 1.6], 
#                     [3.2, 0.2, 1.5]])
#     tile_along_beam(x).eval()
#     array([
#        [ 1.29999995,  2.4000001 ,  1.60000002],
#        [ 1.29999995,  2.4000001 ,  1.60000002],
#        [ 1.29999995,  2.4000001 ,  1.60000002],
#        [ 1.29999995,  2.4000001 ,  1.60000002],
#        [ 1.29999995,  2.4000001 ,  1.60000002],
#        [ 3.20000005,  0.2       ,  1.5       ],
#        [ 3.20000005,  0.2       ,  1.5       ],
#        [ 3.20000005,  0.2       ,  1.5       ],
#        [ 3.20000005,  0.2       ,  1.5       ],
#        [ 3.20000005,  0.2       ,  1.5       ]
#        ], dtype=float32)
#     Args:
#       tensor: a 2-D tensor, [batch_size x T]
#     Return:
#       An [batch_size*beam_size x T] tensor, where each row of the input
#       tensor is copied beam_size times in a row in the output
#     """
#     res = tf.tile(tensor, [1, self.beam_size])
#     res = tf.reshape(res, [-1, tf.shape(tensor)[1]])
#     try:
#       new_first_dim = tensor.get_shape()[0] * self.beam_size
#     except Exception:
#       new_first_dim = None
#     res.set_shape((new_first_dim, tensor.get_shape()[1]))
#     return res

#   def calc_topn(self):
#     #[batch_size, beam_size * (max_len-2)]
#     logprobs = tf.concat(self.logprobs_list, 1)

#     #[batch_size, topn]
#     top_logprobs, indices = tf.nn.top_k(logprobs, self.topn)


#     length = self.beam_size * (self.max_len - 2)
#     indice_offsets = tf.reshape(
#       (tf.range(self.batch_size * length) // length) * length,
#       [self.batch_size, length])

#     indice_offsets = indice_offsets[:,0:self.topn]

#     #[batch_size, max_len(length index) - 2, beam_size, max_len]
#     paths = tf.concat(self.paths_list, 1)
#     #top_paths = paths[:, 0:self.topn, 2, :]
    
#     #paths = tf.reshape(paths, [-1, (self.max_len  - 2) * self.beam_size, self.max_len])
#     #paths = tf.reshape(paths, [self.batch_size, -1, self.max_len])

#     paths = tf.reshape(paths, [-1, self.max_len])

#     top_paths = tf.gather(paths, indice_offsets + indices)

#     return top_paths, top_logprobs

#   def take_first_step(self, prev, i, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
#     output_projection = self.output_projection
#     if output_projection is not None:
#       #[batch_size * beam_size, num_units] -> [batch_size * beam_size, num_classes]
#       output = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
#     else:
#       output = prev

#     self.output = output

#     #[batch_size * beam_size, num_classes], here use log sofmax
#     logprobs = tf.nn.log_softmax(output)
    
#     if self.num_classes is None:
#       self.num_classes = tf.shape(logprobs)[1]

#     #->[batch_size, beam_size, num_classes]
#     logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.num_classes])
#     logprobs_batched.set_shape((None, self.beam_size, None))
    
#     # Note: masking out entries to -inf plays poorly with top_k, so just subtract out a large number.
#     nondone_mask = tf.reshape(
#         tf.cast(tf.equal(tf.range(self.num_classes), self.done_token), tf.float32) * -1e18,
#         [1, 1, self.num_classes])

#     #[batch_size, beam_size, num_classes] -> [batch_size, num_classes]
#     #-> past_logprobs[batch_size, beam_size], indices[batch_size, beam_size]
#     past_logprobs, indices = tf.nn.top_k(
#         (logprobs_batched + nondone_mask)[:,0,:],
#         self.beam_size)        

#     # For continuing to the next symbols [batch_size, beam_size]
#     symbols = indices % self.num_classes
#     #from wich beam it comes  [batch_size, beam_size]
#     parent_refs = indices // self.num_classes
    
#     #here when i == 1
#     #[batch_size, beam_size] -> [batch_size, beam_size, 1]
#     #past_symbols = tf.expand_dims(symbols, 2)
#     past_symbols.write(i - 1, symbols)
#     # NOTE: outputing a zero-length sequence is not supported for simplicity reasons

#     #->[batch_size * beam_size,]
#     symbols_flat = tf.reshape(symbols, [-1])

#     return symbols_flat, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams
   
#   def take_step(self, prev, i, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
#     output_projection = self.output_projection
#     if output_projection is not None:
#       #[batch_size * beam_size, num_units] -> [batch_size * beam_size, num_classes]
#       output = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
#     else:
#       output = prev

#     self.output = output

#     #[batch_size * beam_size, num_classes], here use log sofmax
#     logprobs = tf.nn.log_softmax(output)
    
#     if self.num_classes is None:
#       self.num_classes = tf.shape(logprobs)[1]

#     #->[batch_size, beam_size, num_classes]
#     logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.num_classes])
#     logprobs_batched.set_shape((None, self.beam_size, None))
    
#     # Note: masking out entries to -inf plays poorly with top_k, so just subtract out a large number.
#     nondone_mask = tf.reshape(
#         tf.cast(tf.equal(tf.range(self.num_classes), self.done_token), tf.float32) * -1e18,
#         [1, 1, self.num_classes])

#     #logprobs_batched [batch_size, beam_size, num_classes] -> [batch_size, beam_size, num_classes]  
#     #past_logprobs    [batch_size, beam_size] -> [batch_size, beam_size, 1]
#     logprobs_batched = logprobs_batched + tf.expand_dims(past_logprobs, 2)

#     past_logprobs, indices = tf.nn.top_k(
#         tf.reshape(logprobs_batched + nondone_mask, [-1, self.beam_size * self.num_classes]),
#         self.beam_size)

#     # For continuing to the next symbols [batch_size, beam_size]
#     symbols = indices % self.num_classes
#     #from wich beam it comes  [batch_size, beam_size]
#     parent_refs = indices // self.num_classes
    
#     #/notebooks/mine/ipynotebook/tensorflow/beam-search2.ipynb below for mergeing path
#     #here when i >= 2
#     # tf.reshape(
#     #           (tf.range(3 * 5) // 5) * 5,
#     #           [3, 5]
#     #       ).eval()
#     # array([[ 0,  0,  0,  0,  0],
#     #        [ 5,  5,  5,  5,  5],
#     #        [10, 10, 10, 10, 10]], dtype=int32)
#     parent_refs_offsets = tf.reshape(
#         (tf.range(self.batch_size * self.beam_size) // self.beam_size) * self.beam_size,
#         [self.batch_size, self.beam_size])
    
#     #self.past_symbols [batch_size, beam_size, 1]
#     past_symbols_ = past_symbols.stack()
#     past_symbols_batch_major = tf.reshape(past_symbols_, [-1, i-1])
#     beam_past_symbols = tf.gather(past_symbols_batch_major,
#                               parent_refs + parent_refs_offsets)

#     #[batch_size, bam_size, i] the best beam_size paths until step i
#     #past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2)], 2)
#     past_symbols.write(i - 1, symbols)


#     # For finishing the beam 
#     #[batch_size, beam_size]
#     logprobs_done = logprobs_batched[:,:,self.done_token]

#     done_parent_refs = tf.cast(tf.argmax(logprobs_done, 1), tf.int32)
#     done_parent_refs_offsets = tf.range(self.batch_size) * self.beam_size

#     done_past_symbols = tf.gather(past_symbols_batch_major,
#                                   done_parent_refs + done_parent_refs_offsets)

#     #[batch_size, max_len]
#     symbols_done = tf.concat([done_past_symbols,
#                                  tf.ones_like(done_past_symbols[:,0:1]) * self.done_token,
#                                  tf.tile(tf.zeros_like(done_past_symbols[:,0:1]),
#                                          [1, self.max_len - i])
#                                 ], 1)

#     #print(i, 'cde', symbols_done.get_shape())

#     #[batch_size, beam_size] -> [batch_size,]
#     logprobs_done_max = tf.reduce_max(logprobs_done, 1)
  
#     if self.length_normalization_factor > 0:
#       logprobs_done_max /= i ** self.length_normalization_factor

#     #[batch_size, max_len]
#     finished_beams = tf.where(logprobs_done_max > logprobs_finished_beams,
#                                    symbols_done,
#                                    finished_beams)


#     logprobs_finished_beams = tf.maximum(logprobs_done_max, logprobs_finished_beams)

#     #->[batch_size * beam_size,]
#     symbols_flat = tf.reshape(symbols, [-1])

#     return symbols_flat, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams
  
#   def initialize(self):
#     finished = tf.zeros_like(self.decoder_inputs[0] , dtype=tf.bool)
#     next_inputs = self.decoder_inputs[0]
#     return (finished, next_inputs, self.past_symbols, self.past_logprobs, self.finished_beams, self.logprobs_finished_beams)

#   def sample(self, time, outputs, state, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):

#     next_input_id, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams = control_flow_ops.cond(
#       math_ops.equal(time, 1),
#       lambda: self.take_first_step(outputs, time, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams),
#       lambda: self.take_step(outputs, time, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)
#       )
     
#     next_inputs = array_ops.gather(self.embeddings, next_input_id)

#     finished = math_ops.equal(next_input_id, self.done_token)
#     finished = control_flow_ops.cond(math_ops.equal(time, self.max_len),
#           lambda: array_ops.ones([self.batch_size,], dtype=dtypes.bool),
#           lambda: finished)
    
#     return (finished, next_inputs, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams) 


# import tensorflow as tf
# def dynamic_beam_decode(input, max_steps, initial_state, cell, embedding, scope=None,
#                 beam_size=7, done_token=0, num_classes=None,
#                 output_projection=None,  length_normalization_factor=1.0,
#                 prob_as_score=True, topn=1):
#     """
#     Beam search decoder
#     copy from https://gist.github.com/igormq/000add00702f09029ea4c30eba976e0a
#     make small modifications, add more comments and add topn support, and 
#     length_normalization_factor
    
#     Args:
#       decoder_inputs: A list of 2D Tensors [batch_size x input_size].
#       initial_state: 2D Tensor with shape [batch_size x cell.state_size].
#       cell: rnn_cell.RNNCell defining the cell function and size.
#       loop_function: This function will be applied to the i-th output
#         in order to generate the i+1-st input, and decoder_inputs will be ignored,
#         except for the first element ("GO" symbol). 
#         Signature -- loop_function(prev_symbol, i) = next
#           * prev_symbol is a 1D Tensor of shape [batch_size*beam_size]
#           * i is an integer, the step number (when advanced control is needed),
#           * next is a 2D Tensor of shape [batch_size*beam_size, input_size].
#       scope: Passed to seq2seq.rnn_decoder
#       beam_size: An integer beam size to use for each example
#       done_token: An integer token that specifies the STOP symbol
      
#     Return:
#       A tensor of dimensions [batch_size, len(decoder_inputs)] that corresponds to
#       the 1-best beam for each batch.
      
#     Known limitations:
#       * The output sequence consisting of only a STOP symbol is not considered
#         (zero-length sequences are not very useful, so this wasn't implemented)
#       * The computation graph this creates is messy and not very well-optimized

#     TODO: allow return top n result
#     """

#     #TODO may be not input num_classes?
#     assert num_classes is not None

#     sampler = DynamicBeamSampler(input, embedding, max_steps, initial_state, 
#                           done_token=done_token, num_classes=num_classes,
#                           output_projection=output_projection,
#                           length_normalization_factor=length_normalization_factor,
#                           topn=topn)
    
#     decoder = BasicSamplingDecoder(
#         cell=cell,
#         sampler=sampler,
#         initial_state=initial_state,
#         scope=scope)

#     final_outputs, final_state, path, score = dynamic_decode_rnn_beam_search(decoder)

#     if prob_as_score:
#       score = tf.exp(score)
#     return path, score
