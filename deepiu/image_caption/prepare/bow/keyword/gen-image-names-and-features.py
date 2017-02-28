#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-image-names-and-features.py
#        \author   chenghuige  
#          \date   2016-10-07 21:48:59.093268
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('input', '', '')
flags.DEFINE_string('output_dir', '', '')

import numpy as np 
import conf 
from conf import IMAGE_FEATURE_LEN

import gezi

image_names = []
image_features = []

timer = gezi.Timer('gen-image-names-and-features')

for line in open(FLAGS.input):
  l = line.split()
  image_names.append(l[0])
  image_features.append(np.array([float(x) for x in l[1: 1 + IMAGE_FEATURE_LEN]]))

np.save(FLAGS.output_dir + '/image_names.npy', image_names)
np.save(FLAGS.output_dir + '/image_features.npy', image_features)

timer.print()