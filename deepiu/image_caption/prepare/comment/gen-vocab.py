#!/usr/bin/env python
# ==============================================================================
#          \file   gen-vocab.py
#        \author   chenghuige  
#          \date   2016-07-20 21:14:11.200430
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("most_common", 0, "if > 0 then get vocab with most_common words")
flags.DEFINE_integer("min_count", 0, "if > 0 then cut by min_count")
flags.DEFINE_boolean("add_unknown", True, "treat ignored words as unknow")
flags.DEFINE_boolean("save_count_info", True, "save count info to bin")
flags.DEFINE_string("out_dir", '/tmp/train/', "save count info to bin")

import sys,os

import nowarning
from libword_counter import WordCounter
import conf
from conf import WORDS_SEP

assert FLAGS.most_common > 0 or FLAGS.min_count > 0

counter = WordCounter(
    addUnknown=FLAGS.add_unknown,
    mostCommon=FLAGS.most_common,
    minCount=FLAGS.min_count,
    saveCountInfo=FLAGS.save_count_info)

for line in open(sys.argv[1]):
  l = line.rstrip().split('\t')
  words = l[-1]
  for word in words.split(WORDS_SEP):
    counter.add(word)
  
#counter.most_common(1000)
#counter.min_count(100)
counter.finish()

#identifer = counter.get_identifer()
#id = 100
#print identifer.key(id), identifer.value(id), identifer.value(identifer.key(id)), identifer.id(identifer.key(id)), identifer.size()
counter.save(FLAGS.out_dir + '/vocab.bin', FLAGS.out_dir + '/vocab.txt')

  
