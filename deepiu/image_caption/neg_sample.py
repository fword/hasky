#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   neg_sample.py
#        \author   chenghuige  
#          \date   2016-08-20 12:17:00.150253
#   \Description  
# ==============================================================================

"""
depreciated using tfrecord for neg sample
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

import gezi
#NOTCIE change to use gflags as tf.app.flags will have some problem for 2 more files
#@FIMXE
flags = tf.app.flags
import gflags
flags = gflags
FLAGS = flags.FLAGS

flags.DEFINE_string('texts', '/tmp/train/texts.npy', 'texts bin file for text word ids list')
flags.DEFINE_string('distinct_texts', '/tmp/train/distinct_texts.npy', '')
flags.DEFINE_string('text_strs', '/tmp/train/text_strs.npy', 'texts bin file for text word ids list')
flags.DEFINE_string('disticnt_text_strs', '/tmp/train/distinct_text_strs.npy', 'texts bin file for text word ids list') 

print(FLAGS.distinct_texts)
print(FLAGS.texts)

all_texts = np.load(FLAGS.texts)
#all_texts = np.load('/tmp/train/texts.npy')
num_texts = len(all_texts)
print('num_texts:', num_texts, file=sys.stderr)
#all_distinct_texts = np.load(FLAGS.distinct_texts)
#num_distinct_texts = len(all_distinct_texts)
#print('num_distcint_texts:', num_distinct_texts, file=sys.stderr)

all_text_strs = np.load(FLAGS.text_strs)
#all_text_strs = np.load('/tmp/train/text_strs.npy')
#all_distinct_text_strs = np.load(FLAGS.disticnt_text_strs)

def gen_neg_batch(batch_size, num_negs):
  neg_texts = []
  neg_text_strs = []
  for _ in xrange(batch_size):
    negs = np.random.choice(num_texts, num_negs, replace=False)
    neg_texts.append(np.array([all_texts[neg_id] for neg_id in negs]))
    neg_text_strs.append(np.array([all_text_strs[neg_id] for neg_id in negs]))
  return np.array(neg_texts), np.array(neg_text_strs) 
