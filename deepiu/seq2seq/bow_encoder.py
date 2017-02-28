#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   bow_encoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:32.166732
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

#dynamic_batch_length or fixed_batch_length
#combiner can be sum or mean
#always use mask (asume 0 emb index to be 0 vector)
#but now not implement this, will do exp to show if ok @TODO
#for fixed_batch_length sum and mean should be same so just use sum

flags.DEFINE_string('combiner', 'sum', 'actually sum and mean is the same since will norm in cosine same')
flags.DEFINE_boolean('exclude_zero_index', True, 'wether really exclude the first row(0 index)')

import melt
from deepiu.seq2seq.encoder import Encoder

def embedding_lookup(emb, indexes):
  #with tf.device('/cpu:0'):
  return melt.batch_masked_embedding_lookup(emb,
                                            indexes, 
                                            combiner=FLAGS.combiner, 
                                            exclude_zero_index=FLAGS.exclude_zero_index)


#TODO should add emb_bias?
def encode(sequence, emb):
  return embedding_lookup(emb, sequence)

class BowEncoder(Encoder):
  def __init__(self, is_training=True, is_predict=False):
    super(BowEncoder, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict

  def encode(self, sequence, emb=None):
    if emb is None:
      emb = self.emb
    return encode(sequence, emb)
