#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   vocab.py
#        \author   chenghuige  
#          \date   2016-08-19 20:19:09.138521
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#@TODO----remove this ? only need vocab_size
#@FIXME work around to be safe in virtual env for hadoop, if import not at first will segmentation fault after finishing double free core
import gezi.nowarning
from libword_counter import Vocabulary

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS
#NOTICE move this to useage app code
#flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')
#flags.DEFINE_integer('num_reserved_ids', 1, 'reserve one for pad, so to make unk as 1, diff from pad')

flags.DEFINE_integer('vocab_size', 0, '')


import melt
logging = melt.logging

#from  deepiu.image_caption.conf import NUM_RESERVED_IDS
import conf
from conf import NUM_RESERVED_IDS

vocab = None 
vocab_size = None

#@TODO one big problem is <unk> now just filterd should use it and let 0 be padding
#Also gen vocab must add start <s> and end </s>, but for bow you can ignore them when encoding text
#you have <unk> in rnn method not ignore and may be in bow you ignore it but it should in vocabulary
#and have index > 1, num_reserverd_ids > 0 pad, pad1, pad2 ..., at least to have 0 pad occupied
def get_vocab():
  init()
  return vocab

def get_vocab_size():
  init()
  return vocab_size

def end_id():
  init()
  return vocab.end_id() 

def start_id():
  init()
  return vocab.start_id()

def go_id():
  init()
  return vocab.id('<GO>') 
  #return NUM_RESERVED_IDS - 1
  
def init():
  global vocab, vocab_size
  if vocab is None:
    logging.info('vocab:{}'.format(FLAGS.vocab))
    vocab = Vocabulary(FLAGS.vocab, NUM_RESERVED_IDS)
    vocab_size = vocab.size() if not FLAGS.vocab_size else min(vocab.size(), FLAGS.vocab_size)
    logging.info('vocab_size:{}'.format(vocab_size))
    assert vocab_size > NUM_RESERVED_IDS, 'empty vocab, wrong vocab path? %s'%FLAGS.vocab
    logging.info('vocab_start:{}'.format(vocab.key(vocab.start_id())))
    logging.info('vocab_end:{}'.format(vocab.key(vocab.end_id())))
    logging.info('vocab_unk:{}'.format(vocab.key(vocab.unk_id())))



