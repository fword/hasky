#!/usr/bin/env python
# ==============================================================================
#          \file   gen-test-imgs.py
#        \author   chenghuige  
#          \date   2016-07-26 13:00:37.741672
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dir', '/tmp/train/', '')
flags.DEFINE_string('image_file', '/home/gezi/data/image-auto-comment/other/img2fea.txt', '')
flags.DEFINE_integer('num_examples', 100, '')

import numpy as np

features = []
names = []
for i, line in enumerate(open(FLAGS.image_file)):
  if i == FLAGS.num_examples:
    break
  l = line.split('\t')
  name = l[0]
  feature = [float(x) for x in l[1:]]
  features.append(feature)
  names.append(name)

np.save(FLAGS.dir + '/evaluate_imgs.npy', np.array(features, dtype=np.float32))
np.save(FLAGS.dir + '/evaluate_img_names.npy', np.array(names))

