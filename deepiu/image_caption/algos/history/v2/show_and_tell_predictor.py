#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell_predictor.py
#        \author   chenghuige  
#          \date   2016-09-04 17:50:21.017234
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


import numpy as np

import melt
import conf 
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS

import vocabulary

import text2ids 
from text2ids import idslist2texts

import gezi

from algos.show_and_tell import ShowAndTell

class SeqDecodeMethod():
  max_prob = 0
  sample = 1
  full_sample = 2
  beam_search = 3

class ShowAndTellPredictor(ShowAndTell, melt.PredictorBase):
  def __init__(self):
    #super(ShowAndTellPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    ShowAndTell.__init__(self, is_training=False)

  def init_predict_texts(self, decode_method=0, beam_size=5):
    """
    init for generate texts
    """
    self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image')
    self.texts = self.build_predict_texts_graph(self.image_feature_place, decode_method, beam_size)

  def predict_texts(self, images):
    feed_dict = {
      self.image_feature_place: images,
      }

    vocab = vocabulary.get_vocab()
    generated_words = self.sess.run(self.texts, feed_dict) 
 
    texts = idslist2texts(generated_words)
    return texts

  def init_predict(self):
    self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image')
    self.text = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS])
    self.loss = self.build_predict_graph(self.image_feature_place, self.text)

  def predict(self, image, text):
    """
    default usage is one single image , single text predict one sim score
    """
    feed_dict = {
      self.image_feature_place: image.reshape([-1, IMAGE_FEATURE_LEN]),
      self.text: text.reshape([-1, TEXT_MAX_WORDS]),
    }
    loss = self.sess.run(self.loss, feed_dict)
    return loss

  def bulk_predict(self, images, texts):
    """
    input multiple images, multiple texts
    outupt: 
    
    image0, text0_score, text1_score ...
    image1, text0_score, text1_score ...
    ...

    """
    scores = []
    for image in images:
      stacked_images = np.array([image] * len(texts))
      score = self.predict(stacked_images, texts)
      scores.append(score)
    return np.array(scores)

  def build_predict_texts_graph(self, image, decode_method=0, beam_size=5):
    """
    @TODO beam search, early stop maybe need c++ op
    """
    batch_size = tf.shape(image)[0]

    image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

    state = self.cell.zero_state(batch_size, tf.float32)
 
    generated_words = []

    max_words = TEXT_MAX_WORDS
    with tf.variable_scope("RNN"):
      (output, state) = self.cell(image_emb, state)

      last_word = tf.nn.embedding_lookup(self.emb, tf.zeros([batch_size], tf.int32)) + self.bemb

      #last_word = image_emb
      for i in range(max_words):
        #if i > 0: tf.get_variable_scope().reuse_variables()
        tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(last_word, state)

        with tf.device('/cpu:0'):
          logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b

        top_prob_words = None
        if decode_method == SeqDecodeMethod.max_prob:
          max_prob_word = tf.argmax(logit_words, 1)
        elif decode_method == SeqDecodeMethod.sample:
          max_prob_word = tf.nn.top_k(logit_words, beam_size)[1][:, np.random.choice(beam_size, 1)]
        elif decode_method == SeqDecodeMethod.full_sample:
          top_prob_words = tf.nn.top_k(logit_words, beam_size)[1]
          max_prob_word = top_prob_words[:, np.random.choice(beam_size, 1)]
        elif decode_method == SeqDecodeMethod.beam_search:
          raise ValueError('beam search nor implemented yet')
        else:
          raise ValueError('not supported decode method')

        last_word = tf.nn.embedding_lookup(self.emb, max_prob_word) + self.bemb
        max_prob_word = tf.reshape(max_prob_word, [batch_size, -1])

        if top_prob_words is not None:
          generated_words.append(top_prob_words)
        else:
          generated_words.append(max_prob_word)

      generated_words = tf.concat(1, generated_words)
      return generated_words

  def build_predict_graph(self, image, text):
    image = tf.reshape(image, [1, IMAGE_FEATURE_LEN])
    text = tf.reshape(text, [1, TEXT_MAX_WORDS])
    
    loss = self.build_graph(image, text)
    return loss
