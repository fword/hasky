#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
lstm based generative model

@TODO try to use seq2seq.py 
* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

      TODO may be self_nomralization is better for performance of train and infernce then sampled softmax

keyword state of art
batch_size:[256] batches/s:[8.44] insts/s:[2160.43]
old version on gpu0 machine batches/s:[4.13]
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_neg', False, 'use neg means using hinge loss(rank nce)')
flags.DEFINE_boolean('show_neg', False, 'show neg means show neg score')

import melt
logging = melt.logging
from deepiu.image_caption import conf 
from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS
from deepiu.util import vocabulary
from deepiu.seq2seq import embedding
import deepiu
  
class ShowAndTell(object):
  """
  ShowAndTell class is a trainer class
  but has is_training mark for ShowAndTell predictor will share some code here
  3 modes
  train,
  evaluate,
  predict
  """
  def __init__(self, is_training=True, is_predict=False):
    super(ShowAndTell, self).__init__()

    #---------should be show_and_tell/model_init_1
    #print('ShowAndTell init:', tf.get_variable_scope().name)
    #self.abcd = melt.init_bias(3)
    #print('ShowAndTell bias', self.abcd.name)

    self.is_training = is_training 
    self.is_predict = is_predict

    #if is_training:
    logging.info('num_sampled:{}'.format(FLAGS.num_sampled))
    logging.info('use_neg:{}'.format(FLAGS.use_neg))
    logging.info('num_sampled:{}'.format(FLAGS.num_sampled))
    logging.info('log_uniform_sample:{}'.format(FLAGS.log_uniform_sample))
    logging.info('num_layers:{}'.format(FLAGS.num_layers))
    logging.info('keep_prob:{}'.format(FLAGS.keep_prob))
    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('add_text_start:{}'.format(FLAGS.add_text_start))
    logging.info('zero_as_text_start:{}'.format(FLAGS.zero_as_text_start))

    emb = embedding.get_embedding('emb')
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.decoder = deepiu.seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    emb_dim = FLAGS.emb_dim
    self.encode_img_W = melt.variable.get_weights_uniform(
      'encode_img_W', 
      [IMAGE_FEATURE_LEN, emb_dim], 
      -FLAGS.initializer_scale, 
      FLAGS.initializer_scale)
    self.encode_img_b = melt.variable.get_bias('encode_img_b', [emb_dim])


  def feed_ops(self):
    """
    return feed_ops, feed_run_ops
    same as ptm example code
    not used very much, since not imporve result
    """
    if FLAGS.feed_initial_sate:
      return [self.decoder.initial_state], [self.decoder.final_state]
    else:
      return [], []

  def compute_seq_loss(self, image_emb, text):
    return self.decoder.sequence_loss(image_emb, text)

  #NOTICE mainly usage is not use neg! for generative method
  def build_graph(self, image_feature, text, neg_text=None):
    image_emb = tf.nn.xw_plus_b(image_feature, self.encode_img_W, self.encode_img_b)
    pos_loss = self.compute_seq_loss(image_emb, text)

    loss = None
    scores = None
    if neg_text is not None and (FLAGS.use_neg or FLAGS.show_neg):
      neg_losses = []
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        tf.get_variable_scope().reuse_variables()
        neg_text_i = neg_text[:, i, :]
        neg_loss = self.compute_seq_loss(image_emb, neg_text_i)
        neg_losses.append(neg_loss)

      neg_losses = tf.concat(1, neg_losses)
      
      if FLAGS.use_neg:
        #neg_losses [batch_size, num_neg_text] 
        loss = melt.hinge_loss(-pos_loss, -neg_losses, FLAGS.margin)
      
      scores = tf.concat(1, [pos_loss, neg_losses])

    if loss is None:
      if not self.is_predict:
        loss = tf.reduce_mean(pos_loss)
      else:
        loss = pos_loss
      
      if scores is None:
        if neg_text is not None:
          scores = tf.concat(1, [pos_loss, pos_loss])
        else:
          scores = pos_loss

    tf.add_to_collection('scores', scores)
    return loss

  def build_train_graph(self, image_feature, text, neg_text=None):
    return self.build_graph(image_feature, text, neg_text)
