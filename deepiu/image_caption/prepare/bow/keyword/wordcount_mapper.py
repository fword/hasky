#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   wordcount_mapper.py
#        \author   chenghuige  
#          \date   2016-10-09 00:59:56.649588
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('seg_method', 'default', '')
import sys,os
import melt

import conf 
from conf import IMAGE_FEATURE_LEN

from gezi import Segmentor
segmentor = Segmentor()

num = 0
total_count = 0
for line in sys.stdin:
  if num % 10000 == 0:
    print(num, file=sys.stderr)
  l = line.rstrip().split('\t')
  img_end = IMAGE_FEATURE_LEN + 1
  texts = [x.split('\x01')[0] for x in l[img_end:]]
  for text in texts:
    words = segmentor.Segment(text, FLAGS.seg_method)
    for word in words:
      print('%s\t%d' % (word, 1))
      total_count += 1
  num += 1

print('<TotalCount>\t%d' % total_count)
