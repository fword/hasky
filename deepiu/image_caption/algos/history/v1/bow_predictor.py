#!/usr/bin/env python
# ==============================================================================
#          \file   bow_predictor.py
#        \author   chenghuige  
#          \date   2016-08-18 00:50:13.151993
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

from algos.bow import Bow

#@TODO used in placholder , better handle ?

import conf
from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS, MAX_EMB_WORDS


class BowPredictor(Bow, melt.PredictorBase):
  """
  @NOTICE dot not use BowPredictor directly
  use as
  predictor = algos.algos_factor.gen_predictor('bow')
  or 
  with tf.variable_scope('model_init'):
    predictor = BowPredictor()
  By adding model_init scope, we can use Bow and BowPredictor at same time
  """
  def __init__(self):
    #super(BowPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    Bow.__init__(self, False)

    self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image_feature') 
    self.text_place = tf.placeholder(tf.int32, [None, TEXT_MAX_WORDS], name='text')

  def init_predict(self):
    self.score = self.build_predict_graph()

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

  def build_graph(self, image_feature, text):
    with tf.variable_scope("image_text_sim"):
      image_feature = self.forward_image_feature(image_feature)
      text_feature = self.forward_text(text)
      score = melt.cosine(image_feature, text_feature, nonorm=True)
      return score

  def build_predict_graph(self):
    score = self.build_graph(self.image_feature_place, self.text_place)
    return score

  def build_fixed_text_graph(self, text_feature_npy): 
    """
    text features directly load to graph, @NOTICE text_feature_npy all vector must of same length
    used in evaluate.py for both fixed text and fixed words
    """
    with tf.variable_scope("image_text_sim"):
      image_feature = self.forward_image_feature(self.image_feature_place)
      text_feature = melt.constant(self.sess, text_feature_npy)
      score = melt.cosine(image_feature, text_feature, nonorm=True)
      return score

  #--------- only used during training evaluaion, image_feature and text all small
  def build_train_graph(self, image_feature, text, neg_text):
    """
    Only used for train and evaluation, hack!
    """
    return super(BowPredictor, self).build_graph(image_feature, text, neg_text)
  
  def init_evaluate_constant_image(self, image_feature_npy):
    self.image_feature = tf.constant(image_feature_npy)

  def init_evaluate_constant_text(self, text_npy):
    #self.text = tf.constant(text_npy)
    self.text = melt.constant(self.sess, text_npy)

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
    #or to put on cpu ?
    # du -h comment_feature_final.npy 
    #3.6G	comment_feature_final.npy  so 100w 4G, 800w 32G, 1500w word will exceed cpu 
   
    #emb = melt.first_nrows(self.emb, MAX_EMB_WORDS)
    #should not larger then vocab_zie
    word_emb = self.emb if self.vocab_size <= MAX_EMB_WORDS else self.emb[:MAX_EMB_WORDS,:]
    if not FLAGS.dynamic_batch_length and not FLAGS.exclude_zero_index:
      #combiner under this condition must be sum, bow.py __init__ checked this
      word_emb = (word_emb + tf.nn.embedding_lookup(word_emb, [0]) * (TEXT_MAX_WORDS - 1))
    word_feature = self.forward_text_feature(word_emb)
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
