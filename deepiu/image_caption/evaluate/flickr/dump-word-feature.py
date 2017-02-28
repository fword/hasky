#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-word-feature.py
#        \author   chenghuige  
#          \date   2016-08-21 21:48:41.457382
#   \Description  
# ==============================================================================
  
"""
should be used after prepare/gen-distinct-texts.py
and train.py
with model ready
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './model', '')  
flags.DEFINE_string('data_dir', '/tmp/train', '')  

import numpy as np
from predictor import Predictor
import melt

def dump():
  predictor = Predictor()
  tf_word_feature = predictor.forward_allword()
  sess = melt.load(FLAGS.model_dir)
  word_feature = sess.run(tf_word_feature)
  np.save(FLAGS.data_dir + '/word_feature.npy', word_feature)
  sess.close()

def main(_):
  dump()

if __name__ == '__main__':
  tf.app.run()