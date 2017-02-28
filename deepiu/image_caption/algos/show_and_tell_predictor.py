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

import gezi
import melt

from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS
from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts
from deepiu.image_caption.algos.show_and_tell import ShowAndTell

from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

class ShowAndTellPredictor(ShowAndTell, melt.PredictorBase):
  def __init__(self):
    #super(ShowAndTellPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    ShowAndTell.__init__(self, is_training=False, is_predict=True)

    self.text_list = []
    ##show show_and_tell/model_init
    #print('ShowAndTellPredictor init:', tf.get_variable_scope().name)
    self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image_feature')
    ##show  show_and_tell/model_init_1
    #print('image_feature_place', self.image_feature_place)

  def restore_from_graph(self):
    """
    depreciated, juse use melt.Predictor 
    if want to restore from graph directly without building predict graph
    """
    self.text_list = list(tf.get_collection('text')[0], tf.get_collection('text_beam')[0])
    ##tensorflow.python.framework.errors.InvalidArgumentError: You must feed a value for placeholder tensor 'show_and_tell/model_init_1/image' with dtype float
    ##ValueError: Name 'show_and_tell/model_init_1/image:0' appears to refer to a Tensor, not a Operation.
    #self.image_feature_place = tf.get_default_graph().get_operation_by_name('show_and_tell/model_init_1/image:0').outputs[0]
    ##placeholder should treat as below, refer to models/image/imagenet of tf
    self.image_feature_place = 'show_and_tell/model_init_1/image:0'

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    #@NOTICE if you want differnt graph then self.image_feature_place must in __init__(outside) not here
    #because it is a shared place holder, if here not, even reuse variable, 
    #will generate more then 1 place holder, reuse not for placeholder
    text = self.build_predict_text_graph(self.image_feature_place, 
      decode_method, 
      beam_size, 
      convert_unk)

    self.text_list.append(text)

    return text


  def predict_text(self, images, index=0):
    feed_dict = {
      self.image_feature_place: images,
      }

    vocab = vocabulary.get_vocab()

    generated_words = self.sess.run(self.text_list[index], feed_dict) 
    texts = idslist2texts(generated_words)

    return texts

  def init_predict(self, text_max_words=TEXT_MAX_WORDS):
    self.text_place = tf.placeholder(tf.int64, [None, text_max_words], name='text')
    # TODO try make gpu mem ok
    #with tf.device('/cpu:0'):
    self.score = self.build_predict_graph(self.image_feature_place, self.text_place, text_max_words)
    return self.score

  def predict(self, image, text):
    """
    default usage is one single image , single text predict one sim score
    """
    feed_dict = {
      self.image_feature_place: image.reshape([-1, IMAGE_FEATURE_LEN]),
      self.text_place: text.reshape([-1, TEXT_MAX_WORDS]),
    }
    score = self.sess.run(self.score, feed_dict)
    score = score.reshape((len(text),))
    return score

  # def _bulk_predict(self, image, texts):
  #   stacked_images = np.array([image] * len(texts))
  #   score = self.predict(stacked_images, texts)

  #TODO why use much mem ? when texts num is large ?
  #def bulk_predict(self, images, texts, batch_size=100):
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
 
  def build_predict_text_graph(self, image, decode_method=0, beam_size=5, convert_unk=True):
    """
    @TODO beam search, early stop maybe need c++ op
    """
    image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

    decoder_input = image_emb
    state = None
    
    if decode_method != SeqDecodeMethod.beam_search:
      return self.decoder.generate_sequence(decoder_input, TEXT_MAX_WORDS, state, decode_method, convert_unk)
    else:
      return self.decoder.generate_sequence_by_beam_search(decoder_input, 
                                                           TEXT_MAX_WORDS, 
                                                           state, 
                                                           beam_size, 
                                                           convert_unk,
                                                           length_normalization_factor=0.)


  def build_predict_graph(self, image, text, text_max_words=TEXT_MAX_WORDS):
    image = tf.reshape(image, [-1, IMAGE_FEATURE_LEN])
    text = tf.reshape(text, [-1, text_max_words])
    
    loss = self.build_graph(image, text)
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score
