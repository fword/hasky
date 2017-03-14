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
    self.image_feature_place = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image_feature')
    self.text_place = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text = self.build_predict_text_graph(self.image_feature_place, 
      decode_method, 
      beam_size, 
      convert_unk)

    self.text_list.append(text)

    return text

  def init_predict(self, exact_loss=False):
    self.score = self.build_predict_graph(self.image_feature_place, 
                                          self.text_place, 
                                          exact_loss=exact_loss)
    return self.score
 
  def build_predict_text_graph(self, image, decode_method=0, beam_size=5, convert_unk=True):
    """
    @TODO beam search, early stop maybe need c++ op
    """
    image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

    decoder_input = image_emb
    state = None
    
    if decode_method == SeqDecodeMethod.greedy:
      return self.decoder.generate_sequence_greedy(decoder_input, 
                                            max_steps=TEXT_MAX_WORDS, 
                                            initial_state=state, 
                                            convert_unk=convert_unk)
    elif decode_method == SeqDecodeMethod.beam:
      return self.decoder.generate_sequence_beam(decoder_input,
                                                 max_steps=TEXT_MAX_WORDS, 
                                                 initial_state=state, 
                                                 beam_size=beam_size, 
                                                 convert_unk=convert_unk,
                                                 length_normalization_factor=0.)
    else:
      raise ValueError('not supported decode_method: %d' % decode_method)

  def build_predict_graph(self, image, text, exact_loss=False):
    image = tf.reshape(image, [-1, IMAGE_FEATURE_LEN])
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
    
    loss = self.build_graph(image, text)
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score

  #--------------below depreciated, just use melt.predictor for inference
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

  def predict_text(self, images, index=0):
    """
    depreciated will remove
    """
    feed_dict = {
      self.image_feature_place: images,
      }

    vocab = vocabulary.get_vocab()

    generated_words = self.sess.run(self.text_list[index], feed_dict) 
    texts = idslist2texts(generated_words)

    return texts