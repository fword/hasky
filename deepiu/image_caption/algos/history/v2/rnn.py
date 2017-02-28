#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2016-09-22 19:13:30.275142
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  

import melt
import conf 
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS
import vocabulary
from algos.bow import Bow

cell = tf.nn.rnn_cell.BasicLSTMCell
#cell = tf.nn.rnn_cell.GRUCell

class Rnn(Bow):
  def __init__(self, is_training=True):
    #super(Rnn, self).__init__()
    Bow.__init__(self, is_training=is_training)
    self.is_training = is_training

    self.end_id = self.vocab_size - 1
    hidden_size = FLAGS.hidden_size

    self.cell = cell(hidden_size, state_is_tuple=True)
    if is_training and FLAGS.keep_prob < 1:
     self.cell = tf.nn.rnn_cell.DropoutWrapper(
         self.cell, output_keep_prob=FLAGS.keep_prob)
    self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * FLAGS.num_layers, state_is_tuple=True)

  def gen_text_feature(self, text):
    is_training = self.is_training
    batch_size = tf.shape(text)[0]
    
    zero_pad = tf.zeros([batch_size, 1], dtype=text.dtype)
    text = tf.concat(1, [zero_pad, text, zero_pad])
    sequence_length = melt.length(text) + 1
    text = melt.dynamic_append_with_length(text, sequence_length, tf.constant(self.end_id, dtype=text.dtype)) 
    sequence_length += 1

    state = self.cell.zero_state(batch_size, tf.float32)

    inputs = tf.nn.embedding_lookup(self.emb, text) 
    if is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

    outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=state, 
                                       sequence_length=sequence_length)

    text_feature = melt.dynamic_last_relevant(outputs, sequence_length)
    return text_feature

  def build_train_graph(self, image_feature, text, neg_text, lookup_negs_once=False):
    return self.build_graph(image_feature, text, neg_text, lookup_negs_once=lookup_negs_once)
   