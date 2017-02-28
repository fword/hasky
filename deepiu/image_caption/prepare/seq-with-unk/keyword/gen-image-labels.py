#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-image-labels.py
#        \author   chenghuige  
#          \date   2016-10-07 20:41:05.624073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('input', '', '')
flags.DEFINE_string('output', '', '')

import numpy as np 
import conf 
from conf import IMAGE_FEATURE_LEN

image_labels = {}

print('input:', FLAGS.input)
print('output:', FLAGS.output)

for line in open(FLAGS.input):
  l = line.rstrip().split('\t')
  image = l[0]
  img_end = IMAGE_FEATURE_LEN + 1
  if image not in image_labels:
    image_labels[image] = set()
  m = image_labels[image]
  texts = [x.split('\x01')[0] for x in l[img_end:]]
  for text in texts:
    m.add(text)

np.save(FLAGS.output, image_labels)
