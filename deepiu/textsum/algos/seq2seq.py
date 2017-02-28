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

  def build_graph(self, input_text, text):
    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text)
    with tf.variable_scope("decode"):
      #loss = self.decoder.sequence_loss(encoder_output, text, state)
      #loss = self.decoder.sequence_loss(encoder_output, text, None)
      if not FLAGS.use_attention:
        loss = self.decoder.sequence_loss(None, text, state)
      else:
        loss = self.decoder.sequence_loss_with_attention(None, text, state, encoder_output)
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

    self.text_list = []
    self.input_text_place = tf.placeholder(tf.int64, [None, INPUT_TEXT_MAX_WORDS], name='input_text')


  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    #@NOTICE if you want differnt graph then self.image_feature_place must in __init__(outside) not here
    #because it is a shared place holder, if here not, even reuse variable, 
    #will generate more then 1 place holder, reuse not for placeholder
    text, score = self.build_predict_text_graph(self.input_text_place, 
      decode_method, 
      beam_size, 
      convert_unk)

    self.text_list.append(text)

    return text, score


  def predict_text(self, input_texts, index=0):
    feed_dict = {
      self.input_text_place: input_texts,
      }

    vocab = vocabulary.get_vocab()

    generated_words = self.sess.run(self.text_list[index], feed_dict) 
    texts = idslist2texts(generated_words)

    return texts

  def init_predict(self, input_text_max_words=INPUT_TEXT_MAX_WORDS, text_max_words=TEXT_MAX_WORDS):
    self.text_place = tf.placeholder(tf.int64, [None, text_max_words], name='text')
    self.score = self.build_predict_graph(self.input_text_place, self.text_place, 
                                          input_text_max_words, text_max_words)
    return self.score

  def predict(self, input_text, text):
    """
    default usage is one single input text, single text predict one sim score
    """
    feed_dict = {
      self.input_text_place: input_text.reshape([-1, INPUT_TEXT_MAX_WORDS]),
      self.text_place: text.reshape([-1, TEXT_MAX_WORDS]),
    }
    score = self.sess.run(self.score, feed_dict)
    score = score.reshape((len(text),))
    return score

  #TODO why use much mem ? when texts num is large ?
  #def bulk_predict(self, images, texts, batch_size=100):
  def bulk_predict(self, input_texts, texts):
    """
    input multiple input texts, multiple texts
    outupt: 
    
    image0, text0_score, text1_score ...
    image1, text0_score, text1_score ...
    ...
    """
    scores = []
    for input_text in input_texts:
      stacked_input_texts = np.array([input_text] * len(texts))
      score = self.predict(stacked_input_texts, texts)
      scores.append(score)
    return np.array(scores)
 
  def build_predict_text_graph(self, input_text, decode_method=0, beam_size=5, convert_unk=True):
    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text)
    with tf.variable_scope("decode"):
      #return self.decoder.generate_sequence(encoder_output, TEXT_MAX_WORDS, state, decode_method, beam_size, convert_unk)
      #return self.decoder.generate_sequence(encoder_output, TEXT_MAX_WORDS, None, decode_method, beam_size, convert_unk)
      decoder_input = self.decoder.get_start_embedding_input(encoder_output)

      if not FLAGS.use_attention:
        if decode_method != SeqDecodeMethod.beam_search or not FLAGS.use_beam_search:
          return self.decoder.generate_sequence(decoder_input, TEXT_MAX_WORDS, state, decode_method, convert_unk)
        else:
          return self.decoder.generate_sequence_by_beam_search(decoder_input, 
                                                               TEXT_MAX_WORDS, 
                                                               state, 
                                                               beam_size, 
                                                               convert_unk,
                                                               length_normalization_factor=0.)
      else:
        #TODO: add beam search support
        if decode_method != SeqDecodeMethod.beam_search or not FLAGS.use_beam_search:
          return self.decoder.generate_sequence_with_attention(decoder_input, TEXT_MAX_WORDS, state, decode_method, convert_unk)
        else:
          return self.decoder.generate_sequence_by_beam_search(decoder_input, 
                                                               TEXT_MAX_WORDS, 
                                                               state, 
                                                               beam_size, 
                                                               convert_unk,
                                                               length_normalization_factor=0.)

  def build_predict_graph(self, input_text, text, input_text_max_words=INPUT_TEXT_MAX_WORDS, text_max_words=TEXT_MAX_WORDS):
    input_text = tf.reshape(input_text, [-1, input_text_max_words])
    text = tf.reshape(text, [-1, text_max_words])
    
    loss = self.build_graph(input_text, text)
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score
