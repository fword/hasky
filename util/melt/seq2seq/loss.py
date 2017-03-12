#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2017-01-16 14:14:45.444355
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
  
#-----------------------loss 
# if use tf.contrib.seq2seq.sequence_loss(which is per example if set average_across_batch=False) 
# notice the main diff here is we do  targets = array_ops.reshape(targets, [-1, 1]) 
#so in sample soft max  you do not need to do reshape  labels = tf.reshape(labels, [-1, 1])
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
    logits_shape = array_ops.shape(logits)
    batch_size = logits_shape[0]
    if softmax_loss_function is None:
      #croosents [batch_size, num_steps]
      num_classes = logits_shape[-1]
      logits = array_ops.reshape(logits, [-1, num_classes])
      targets = array_ops.reshape(targets, [-1])
      crossents = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    else:
      emb_dim = logits_shape[-1]
      #need reshape because unlike sparse_softmax_cross_entropy_with_logits, 
      #tf.nn.sampled_softmax_loss now only accept 2d [batch_size, dim] as logits input
      logits = array_ops.reshape(logits, [-1, emb_dim])
      targets = array_ops.reshape(targets, [-1, 1])
      #croosents [batch_size * num_steps]
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

import tensorflow as tf  
import melt
#sample loss must be None
def exact_predict_loss(logits, targets, mask, num_steps, batch_size=None):
  if batch_size is None:
    batch_size = tf.shape(logits)[0]
  i = tf.constant(0, dtype=tf.int32)
  condition = lambda i, log_probs: tf.less(i, num_steps)
  log_probs = tf.zeros([batch_size,], dtype=tf.float32)
  def body(i, log_probs):
    step_logits = logits[:, i, :]
    step_probs = tf.nn.softmax(step_logits)
    step_targets = targets[:, i]
    selected_probs = melt.dynamic_gather2d(step_probs, step_targets)
    selected_log_probs = tf.log(tf.maximum(selected_probs, 1e-12))
    step_mask = mask[:, i]
    log_probs += selected_log_probs * step_mask
    return tf.add(i, 1), tf.reshape(log_probs, [batch_size,])
  _, log_probs = tf.while_loop(condition, body, [i, log_probs])
  loss = -log_probs;
  return loss


def gen_sampled_softmax_loss_function(num_sampled, vocab_size, 
                                      weights, biases,
                                      log_uniform_sample=True,
                                      is_predict=False,
                                      sample_seed=None,
                                      vocabulary=None):
  #@TODO move to melt  def prepare_sampled_softmax_loss(num_sampled, vocab_size, emb_dim)
  #return output_projection, softmax_loss_function
  #also consider candidate sampler 
  # Sampled softmax only makes sense if we sample less than vocabulary size.
  # FIMXE seems not work when using meta graph load predic cause segmentaion fault if add not is_predict
  #if not is_predict and (num_sampled > 0 and num_sampled < vocab_size):
  if num_sampled > 0 and num_sampled < vocab_size:
    def sampled_loss(inputs, labels):
      #with tf.device("/cpu:0"):
      #---use this since default tf.contrib.seq2seq input labels as [-1], if you use melt.seq2seq.sequence_loss do not need reshape
      #labels = tf.reshape(labels, [-1, 1])

      if log_uniform_sample:
        #most likely go here
        sampled_values = None
        if is_predict:
          sampled_values = tf.nn.log_uniform_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocab_size,
            seed=sample_seed)
      else:
        vocab_counts_list = [vocabulary.vocab.freq(i) for i in xrange(vocab_size)]
        vocab_counts_list[0] = 1  #  --hack..
        #vocab_counts_list = [vocabulary.vocab.freq(i) for i in xrange(NUM_RESERVED_IDS, vocab_size)
        #-------above is ok, do not go here
        #@TODO find which is better log uniform or unigram sample, what's the diff with full vocab
        #num_reserved_ids: Optionally some reserved IDs can be added in the range
        #`[0, num_reserved_ids]` by the users. One use case is that a special
        #unknown word token is used as ID 0. These IDs will have a sampling
        #probability of 0.
        #@TODO seems juse set num_reserved_ids to 1 to ignore pad, will we need to set to 2 to also ignore UNK?
        #range_max: An `int` that is `>= 1`.
        #The sampler will sample integers from the interval [0, range_max).

        ##---@FIXME setting num_reserved_ids=NUM_RESERVED_IDS will cause Nan, why...
        #sampled_values = tf.nn.fixed_unigram_candidate_sampler(
        #    true_classes=labels,
        #    num_true=1,
        #    num_sampled=num_sampled,
        #    unique=True,
        #    range_max=vocab_size - NUM_RESERVED_IDS,
        #    distortion=0.75,
        #    num_reserved_ids=NUM_RESERVED_IDS,
        #    unigrams=vocab_counts_list)
        
        #FIXME keyword app, now tested on basice 50w,
        # sample 1000 or other num will cause nan use sampled by count or not by count
        #flickr data is ok
        # IMPORTANT to find out reason, data dependent right now
        #seems ok if using dynamic_batch=1, but seems no speed gain... only 5 instance/s
        #NOTICE 50w keyword very slow with sample(1w) not speed gain
        #4GPU will outof mem FIXME  https://github.com/tensorflow/tensorflow/pull/4270
        #https://github.com/tensorflow/tensorflow/issues/4138
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocab_size,
            distortion=0.75,
            num_reserved_ids=0,
            unigrams=vocab_counts_list)

      return tf.nn.sampled_softmax_loss(weights, 
                                        biases, 
                                        labels=labels, 
                                        inputs=inputs, 
                                        num_sampled=num_sampled, 
                                        num_classes=vocab_size,
                                        sampled_values=sampled_values)
                               
    return sampled_loss
  else:
    return None
