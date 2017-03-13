#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   seq2seq.py
#        \author   chenghuige  
#          \date   2016-12-22 20:07:46.827344
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_beam_search', True, '')
  
import melt

from deepiu.seq2seq import embedding
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod
from deepiu import seq2seq

#--for Seq2seqPredictor
from deepiu.textsum.conf import INPUT_TEXT_MAX_WORDS, TEXT_MAX_WORDS
from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts

class Seq2seq(object):
  def __init__(self, is_training=True, is_predict=False):
    super(Seq2seq, self).__init__()

    self.is_training = is_training 
    self.is_predict = is_predict

    emb = embedding.get_embedding('emb')
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.encoder = seq2seq.rnn_encoder.RnnEncoder(is_training, is_predict)
    self.encoder.set_embedding(emb)

    #emb2 = embedding.get_embedding('emb2')
    self.decoder = seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    print('start_id', self.decoder.start_id)

    assert FLAGS.add_text_start is True 
    assert self.decoder.start_id is not None

  def build_graph(self, input_text, text, 
                  exact_prob=False, exact_loss=False):
    """
    exact_prob and exact_loss actually do the same thing,
    they only be used on when is_predict is true
    """
    assert not (exact_prob and exact_loss)
    assert not ((not self.is_predict) and (exact_prob or exact_loss))

    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      loss = self.decoder.sequence_loss(None, text, state, 
                                        attention_states=encoder_output,
                                        exact_prob=exact_prob, 
                                        exact_loss=exact_loss)

      #this is used in train.py for evaluate eval_scores = tf.get_collection('scores')[-1]
      #because we want to show loss for each instance
      if not self.is_training and not self.is_predict:
        tf.add_to_collection('scores', loss)
    
    if not self.is_predict:
      loss = tf.reduce_mean(loss)

    return loss

  def build_train_graph(self, input_text, text):
    return self.build_graph(input_text, text)

class Seq2seqPredictor(Seq2seq, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    Seq2seq.__init__(self, is_training=False, is_predict=True)

    self.input_text_place = tf.placeholder(tf.int64, [None, INPUT_TEXT_MAX_WORDS], name='input_text')
    self.text_place = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text, score = self.build_predict_text_graph(self.input_text_place, 
                                                decode_method=decode_method, 
                                                beam_size=beam_size, 
                                                convert_unk=convert_unk)
    return text, score

  def init_predict(self, exact_prob=False, exact_loss=False):
    score = self.build_predict_graph(self.input_text_place, 
                                     self.text_place, 
                                     exact_prob=exact_prob, 
                                     exact_loss=exact_loss)
    return score

 
  def build_predict_text_graph(self, input_text, decode_method=0, beam_size=5, convert_unk=True):
    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      #TODO notice encoder_output here just used to get batch size
      batch_size = tf.shape(input_text)[0]
      decoder_input = self.decoder.get_start_embedding_input(batch_size)

      if decode_method != SeqDecodeMethod.beam_search:
        return self.decoder.generate_sequence(decoder_input, 
                                       max_steps=TEXT_MAX_WORDS, 
                                       initial_state=state,
                                       attention_states=encoder_output,
                                       decode_method=decode_method,
                                       convert_unk=convert_unk)
      else:
        return self.decoder.generate_sequence_by_beam_search(decoder_input, 
                                                       max_steps=TEXT_MAX_WORDS, 
                                                       initial_state=state,
                                                       attention_states=encoder_output,
                                                       beam_size=beam_size, 
                                                       convert_unk=convert_unk,
                                                       length_normalization_factor=FLAGS.length_normalization_factor)

  def build_predict_graph(self, input_text, text, exact_prob=False, exact_loss=False):
    input_text = tf.reshape(input_text, [-1, INPUT_TEXT_MAX_WORDS])
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
  
    #loss is -logprob  
    loss = self.build_graph(input_text, text, 
                            exact_prob=exact_prob, 
                            exact_loss=exact_loss)
 
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score
