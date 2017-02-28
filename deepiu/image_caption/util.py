#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-19 20:19:09.138521
#   \Description  
# ==============================================================================
"""
depreciated 
use vocab.py
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

def gen_neg_batch(batch_size, num_texts, num_negs, all_texts, all_text_strs):
  neg_comments = []
  neg_comment_strs = []
  for _ in xrange(batch_size):
    negs = np.random.choice(num_texts, num_negs, replace = False)
    neg_texts.append(np.array([all_texts[neg_id] for neg_id in negs]))
    neg_text_strs.append(np.array([all_text_strs[neg_id] for neg_id in negs]))
  return np.array(neg_texts), np.array(neg_text_strs) 


import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')

#@TODO----remove this ? only need vocab_size
import nowarning
from libword_counter import Vocabulary

vocab = None 
vocab_size =None

def init():
  global vocab, vocab_size
  if vocab is None:
    print('FLAGS.vocab:', FLAGS.vocab)
    vocab = Vocabulary(FLAGS.vocab)
    vocab_size = vocab.size()
    print('vocab_size:', vocab_size)

def get_vocab_size():
  global vocab, vocab_size
  print('FLAGS.vocab:', FLAGS.vocab)
  print('vocab:', vocab)
  if vocab is None:
    vocab = Vocabulary(FLAGS.vocab)
    vocab_size = vocab.size()
  print('vocab_size:', vocab_size)
  return vocab_size
