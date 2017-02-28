#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   embedding.py
#        \author   chenghuige  
#          \date   2016-12-24 19:55:37.327855
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_integer('emb_dim', 1024, 'embedding dim for each word, notice for rnn bidirectional here should be acutal emb_dim * 2')
#flags.DEFINE_integer('emb_dim', 256, 'embedding dim for each word, notice for rnn bidirectional here should be acutal emb_dim * 2')

flags.DEFINE_float('weight_stddev', 1e-4,  
                                  """weight stddev, 
                                     @Notice if use bias then small stddev like 0.01 might not lead to convergence, 
                                     causing layer weight value always be 0 with random_normal""")
flags.DEFINE_float('initializer_scale', 0.08, 'used for weights initalize using random_uniform, default value 0.08 follow im2txt')

import tensorflow.contrib.slim as slim

import melt

from deepiu.util import vocabulary  

#TODO try l2_regularizer and compare
#weights = slim.variable('weights',
#                             shape=[10, 10, 3 , 3],
#                             initializer=tf.truncated_normal_initializer(stddev=0.1),
#                             regularizer=slim.l2_regularizer(0.05),
#                             device='/CPU:0')
def get_embedding(name='emb'):
  emb_dim = FLAGS.emb_dim
  vocabulary.init()
  vocab_size = vocabulary.get_vocab_size() 
  
  ##NOTICE if using bidirectional rnn then actually emb_dim is emb_dim / 2, because will as last step depth-concatate output fw and bw vectors
  
  init_width = 0.5 / emb_dim
  emb = melt.variable.get_weights_uniform(name, [vocab_size, emb_dim], -init_width, init_width)
  
  #return to above code if this works not better
  #emb = melt.variable.get_weights_truncated(name, [vocab_size, emb_dim], stddev=FLAGS.weight_stddev)
  
  return emb 

def get_embedding_cpu(name='emb'):
  with tf.device('/CPU:0'):
    return get_embedding(name)

