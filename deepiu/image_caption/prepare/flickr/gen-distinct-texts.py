#!/usr/bin/env python
# ==============================================================================
#          \file   gen-distinct-texts.py
#        \author   chenghuige  
#          \date   2016-07-24 09:21:18.388774
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import numpy as np

import tensorflow as tf 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dir', '/tmp/train/', '')

import gezi

texts = np.load(FLAGS.dir + '/texts.npy')
text_strs = np.load(FLAGS.dir + '/text_strs.npy')

distinct_texts = []
distinct_text_strs = []

maxlen = 0
for text in texts:
  if len(text) > maxlen:
    maxlen = len(text)

text_set = set()
for text, text_str in zip(list(texts), list(text_strs)):
  if text_str not in text_set:
    text_set.add(text_str)
    distinct_texts.append(gezi.pad(text, maxlen))
    distinct_text_strs.append(text_str)

print('num ori texts:', len(texts))
print('num distinct texts:', len(distinct_texts))

distinct_texts = np.array(distinct_texts)
distinct_text_strs = np.array(distinct_text_strs)
np.save(FLAGS.dir + '/distinct_texts.npy', distinct_texts)
np.save(FLAGS.dir + '/distinct_text_strs.npy', distinct_text_strs)
