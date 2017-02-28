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

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('seg_method', 'default', '')

import sys,os

import nowarning
from libsegment import *
import conf
from conf import WORDS_SEP
#need ./data ./conf
#Segmentor.Init()

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

sys.path.append('../')
import gezi 
Segmentor = gezi.Segmentor()

for line in open(sys.argv[1]):
  l = line.rstrip().split('\t')
  img = l[0]
  img = img[img.rindex('/') + 1:]
  if len(l) < 3:
    continue
  index = 0
  for comment in l[2:]:
    if len(comment) == 0:
      continue
    #words = Segmentor.Segment(comment, ' ')

    words = WORDS_SEP.join(Segmentor.Segment(comment, FLAGS.seg_method))

    if len(words) == 0:
      continue
    print('{}#{}\t{}\t{}'.format(img, index, comment, words))
    index += 1

  
