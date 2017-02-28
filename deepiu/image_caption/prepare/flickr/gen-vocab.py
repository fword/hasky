#!/usr/bin/env python
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
@TODO could do segment parallel @TODO
now single thread... slow
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("most_common", 0, "if > 0 then get vocab with most_common words")
flags.DEFINE_integer("min_count", 0, "if > 0 then cut by min_count")
flags.DEFINE_boolean("add_unknown", True, "treat ignored words as unknow")
flags.DEFINE_boolean("save_count_info", True, "save count info to bin")
flags.DEFINE_string("out_dir", '/tmp/train/', "save count info to bin")


assert FLAGS.most_common > 0 or FLAGS.min_count > 0
import nowarning
from libword_counter import WordCounter
counter = WordCounter(
    addUnknown=FLAGS.add_unknown,
    mostCommon=FLAGS.most_common,
    minCount=FLAGS.min_count,
    saveCountInfo=FLAGS.save_count_info)

import sys,os
import numpy as np
import melt

import conf 
from conf import IMAGE_FEATURE_LEN

START_WORD = '<S>'
END_WORD = '</S>'

num = 0
for line in open(sys.argv[1]):
  if num % 10000 == 0:
    print(num)
  l = line.rstrip().split('\t')
  text = l[-1].rstrip()
  words = text.split()
  counter.add(START_WORD)
  for word in words:
    counter.add(word)
  counter.add(END_WORD)
  num += 1
  
counter.save(FLAGS.out_dir + '/vocab.bin', FLAGS.out_dir + '/vocab.txt')
