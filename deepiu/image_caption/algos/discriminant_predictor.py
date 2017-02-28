#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   discriminant_predictor.py
#        \author   chenghuige  
#          \date   2016-09-22 22:39:41.260080
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np

import gezi
import melt

#@TODO should not use conf... change to config file parse instead  @FIXME!! this will cause problem in deepiu package
from deepiu.image_caption.conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS, MAX_EMB_WORDS
from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer

class DiscriminantPredictor(DiscriminantTrainer, melt.PredictorBase):
  def __init__(self):
    #super(DiscriminantPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    DiscriminantTrainer.__init__(self, is_training=False, is_predict=True)

    self.text = None
    self.text_place = None
    self.text2_place = None

    self.image_feature = None
    self.image_feature_place = None

    #input image feature, text ids
    self.score = None
    self.textsim_score = None
    #input image feature, assume text ids already load
    self.fixed_text_score = None
    #input image feature, assume text final feature already load
    self.fixed_text_feature_score = None
  
  def init_predict(self, text_max_words=TEXT_MAX_WORDS):
    self.score = self.build_predict_graph(text_max_words)
    self.textsim_score = self.build_textsim_predict_graph(text_max_words)
    return self.score

  def predict(self, image, text):
    feed_dict = {
      self.image_feature_place: image.reshape([1, IMAGE_FEATURE_LEN]),
      self.text_place: text.reshape([1, TEXT_MAX_WORDS])
      }
    score = self.sess.run(self.score, feed_dict)
    return score[0][0]

  def bulk_predict(self, images, texts):
    """
    input images features [m, ] , texts feature [n,] will
    outut [m, n], for each image output n scores for each text  
    """
    feed_dict = {
      self.image_feature_place: images,
      self.text_place: texts
      }
    score = self.sess.run(self.score, feed_dict)
    return score

  def predict_textsim(self, text, text2):
    feed_dict = {
      self.text_place: text.reshape([1, TEXT_MAX_WORDS]),
      self.text2_place: text.reshape([1, TEXT_MAX_WORDS])
      }
    score = self.sess.run(self.textsim_score, feed_dict)
    return score[0][0]

  def bulk_predict_textsim(self, texts, texts2):
    """
    input images features [m, ] , texts feature [n,] will
    outut [m, n], for each image output n scores for each text  
    """
    feed_dict = {
      self.text_place: texts,
      self.text2_place: texts2
      }
    score = self.sess.run(self.textsim_score, feed_dict)
    return score
  
  def get_image_feature_place(self):
   if self.image_feature_place is None:
     self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image_feature')
   return self.image_feature_place

  def get_text_place(self, text_max_words=TEXT_MAX_WORDS):
    if self.text_place is None:
      self.text_place = tf.placeholder(tf.int32, [None, text_max_words], name='text')
    return self.text_place

  def bulk_fixed_text_predict(self, images):
    """
    not useful compare to bulk_predict since performance is similar, no gain
    """
    if self.fixed_text_score is None:
      assert self.text is not None, 'call init_evaluate_constant at first'
      self.fixed_text_score = self.build_evaluate_fixed_text_graph(self.get_image_feature_place())
    return self.sess.run(self.fixed_text_score, feed_dict={self.image_feature_place: images})

  def bulk_fixed_text_feature_predict(self, images):
    """
    this will be fast
    """
    assert self.fixed_text_feature_score is not None, 'call build_fixed_text_feature_graph(self, text_feature_npy) at first'
    return self.sess.run(self.fixed_text_feature_score, feed_dict={self.image_feature_place: images})

  def build_graph(self, image_feature, text):
    with tf.variable_scope("image_text_sim"):
      image_feature = self.forward_image_feature(image_feature)
      text_feature = self.forward_text(text)
      score = melt.cosine(image_feature, text_feature, nonorm=True)
      return score

  # TODO may consider better performance, reomve redudant like sim(a,b) not need to calc sim(b,a)
  def build_textsim_graph(self, text,  text2):
    with tf.variable_scope("image_text_sim"):
      text_feature = self.forward_text(text)
      text_feature2 = self.forward_text(text2)
      score = melt.cosine(text_feature, text_feature2, nonorm=True)
      return score

  def build_predict_graph(self, text_max_words=TEXT_MAX_WORDS):
    score = self.build_graph(self.get_image_feature_place(), self.get_text_place(text_max_words))
    return score

  def build_textsim_predict_graph(self, text_max_words=TEXT_MAX_WORDS):
    self.text2_place = tf.placeholder(tf.int32, [None, text_max_words], name='text2')
    score = self.build_textsim_graph(self.get_text_place(text_max_words), self.text2_place)
    return score

  def build_fixed_text_feature_graph(self, text_feature_npy): 
    """
    text features directly load to graph, @NOTICE text_feature_npy all vector must of same length
    used in evaluate.py for both fixed text and fixed words
    @FIXME dump text feature should change api
    """
    with tf.variable_scope("image_text_sim"):
      image_feature = self.forward_image_feature(self.image_feature_place)
      text_feature = melt.load_constant(self.sess, text_feature_npy)
      score = melt.cosine(image_feature, text_feature, nonorm=True)
      return score

  #def build_fixed_text_graph(self, text_npy): 
  #  self.init_evaluate_constant(text_npy)
  #  score = self.build_evaluate_fixed_image_text_graph(self.get_image_feature_place())
  #  return score

  def build_fixed_text_graph(self, text_npy): 
    return self.build_fixed_text_feature_graph(text_npy)
    
  #--------- only used during training evaluaion, image_feature and text all small
  def build_train_graph(self, image_feature, text, neg_text, lookup_negs_once=False):
    """
    Only used for train and evaluation, hack!
    """
    return super(DiscriminantPredictor, self).build_graph(image_feature, text, neg_text, lookup_negs_once)
  
  def init_evaluate_constant_image(self, image_feature_npy):
    if self.image_feature is None:
      self.image_feature = tf.constant(image_feature_npy)

  def init_evaluate_constant_text(self, text_npy):
    #self.text = tf.constant(text_npy)
    if self.text is None:
      self.text = melt.load_constant(self.sess, text_npy)

  def init_evaluate_constant(self, image_feature_npy, text_npy):
    self.init_evaluate_constant_image(image_feature_npy)
    self.init_evaluate_constant_text(text_npy)

  def build_evaluate_fixed_image_text_graph(self):
    """
    image features and texts directly load to graph
    """
    score = self.build_graph(self.image_feature, self.text)
    return score

  def forward_word_feature(self):
    #@TODO may need melt.first_nrows so as to avoid caclc to many words
    # du -h comment_feature_final.npy 
    #3.6G	comment_feature_final.npy  so 100w 4G, 800w 32G, 1500w word will exceed cpu 
    num_words = min(self.vocab_size - 1, MAX_EMB_WORDS)
    word_index = tf.reshape(tf.range(num_words), [num_words, 1])
    word_feature = self.forward_text(word_index)
    
    return word_feature

  def build_evaluate_image_word_graph(self, image_feature):
    with tf.variable_scope("image_text_sim"):
      image_feature = self.forward_image_feature(image_feature)
      #no need for embedding lookup
      word_feature = self.forward_word_feature()
      score = melt.cosine(image_feature, word_feature, nonorm=True)
      return score

  def build_evluate_fixed_image_word_graph(self):
    score = self.build_evaluate_image_word_graph(self.image_feature)
    return score

  def build_evaluate_fixed_text_graph(self, image_feature): 
    """
    text features directly load to graph,
    used in evaluate.py for both fixed text and fixed words
    """
    score = self.build_graph(image_feature, self.text)
    return score

  #@TODO for evaluate random data, choose image_feature, and use all text to calc score, and show ori_text, predicted text

  #----------for offline dump usage
  def forward_allword(self):
    """
    only consider single word, using it's embedding as represention
    """
    with tf.variable_scope("image_text_sim"):
      return self.forward_word_feature()

  def forward_fixed_text(self, text_npy):
    #text = tf.constant(text_npy)  #smaller than 2G, then ok... 
    #but for safe in more application
    text = melt.constant(self.sess, text_npy)
    with tf.variable_scope("image_text_sim"):
      text_feature = self.forward_text(text)
      return text_feature