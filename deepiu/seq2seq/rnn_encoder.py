#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_encoder.py
#        \author   chenghuige  
#          \date   2016-12-23 23:59:59.013362
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_integer('rnn_method', 0, '0 forward, 1 backward, 2 bidirectional')
flags.DEFINE_integer('rnn_output_method', 0, '0 sumed vec, 1 last vector, 2 first vector, 3 all here first means first to original sequence')

flags.DEFINE_boolean('encode_start_mark', False, """need <S> start mark""")
flags.DEFINE_boolean('encode_end_mark', False, """need </S> end mark""")
flags.DEFINE_string('encoder_end_mark', '</S>', "or <GO> if NUM_RESERVED_IDS >=3, will use id 2  <PAD2> as <GO>, especailly for seq2seq encoding")


flags.DEFINE_integer('rnn_hidden_size', 1024, 'rnn cell state hidden size, flickr used to use emb_dim 256 and rnn_hidden_size 256')

import functools
import melt
logging = melt.logging

from deepiu.util import vocabulary  

from deepiu.seq2seq.encoder import Encoder

class RnnEncoder(Encoder):
  def __init__(self, is_training=True, is_predict=False):
    super(RnnEncoder, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict
    
    vocabulary.init()

    if FLAGS.encoder_end_mark == '</S>':
      self.end_id =  vocabulary.end_id()
    else:
      self.end_id = vocabulary.go_id() #NOTICE NUM_RESERVED_IDS must >= 3 TODO
    assert self.end_id != vocabulary.vocab.unk_id(), 'input vocab generated without end id'
    
    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)

    #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123));
    if FLAGS.rnn_method == melt.rnn.EncodeMethod.bidrectional:
      self.bwcell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113));
    else:
      self.bwcell = None
  
  def encode(self, sequence, emb=None):
    if emb is None:
      emb = self.emb 

    #--for debug
    sequence, sequence_length = melt.pad(sequence, 
                                     start_id=(vocabulary.vocab.start_id() if FLAGS.encode_start_mark else None),
                                     end_id=(self.end_id if FLAGS.encode_end_mark else None))
    
    if self.is_predict:
      #---only need when predict, since train input already dynamic length, NOTICE this will improve speed a lot
      num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
      sequence = tf.slice(sequence, [0,0], [-1, num_steps])   
    
    inputs = tf.nn.embedding_lookup(emb, sequence) 
    if self.is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

    output_method = FLAGS.rnn_output_method 
    encode_feature, state = melt.rnn.encode(
          self.cell, 
          inputs, 
          sequence_length, 
          cell_bw=self.bwcell,
          encode_method=FLAGS.rnn_method,
          output_method=output_method)

    return encode_feature, state
