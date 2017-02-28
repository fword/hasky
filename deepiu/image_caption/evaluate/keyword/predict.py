#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-19 06:54:26.594835
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('image_feature_place', 'image_feature:0', '')
flags.DEFINE_string('text_place', 'text:0', '')

flags.DEFINE_string('algo', 'bow', '')

flags.DEFINE_string('model_dir', './models/bow', '')
flags.DEFINE_string('vocab', './vocab.bin', 'vocabulary binary file')

#flags.DEFINE_string('seg_method', 'default', '')
#flags.DEFINE_string('feed_single', False, '')

import sys, os
import melt
import deepiu

import conf
from conf import IMAGE_FEATURE_LEN

from deepiu.image_caption import text2ids

import numpy as np

def bulk_predict(predictor, images, texts):
  feed_dict = { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images,
                '%s/%s'%(FLAGS.algo, FLAGS.text_place): texts }
  #---setting to None you can get from erorr log what need to feed
  #feed_dict = None
  #feed_dict = { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images }
  scores = predictor.inference('score', feed_dict)
  scores = scores.reshape((len(texts),))
  return scores

def predict(predictor):
  num = 0
  for line in sys.stdin:
    l = line.rstrip().split('\t')
    simid = l[0]

    keywords = l[1 + IMAGE_FEATURE_LEN:]
    ids = text2ids.texts2ids(keywords, FLAGS.seg_method, FLAGS.feed_single)

    feature = l[1: 1 + IMAGE_FEATURE_LEN]
    if FLAGS.algo != 'show_and_tell':
      feature = np.array([[float(x) for x in feature]])
    else:
      feature = np.array([feature] * len(ids))

    scores = bulk_predict(predictor, feature, ids)
    #print(scores.shape, file=sys.stderr)

    for keyword, score in zip(keywords, scores):
      print('\t'.join([simid, keyword, str(score)]))

    if num % 1000 == 0:
      print(num, file=sys.stderr)
    num += 1

def main(_):
  text2ids.init()
  predictor = melt.Predictor(FLAGS.model_dir)
  predict(predictor)

if __name__ == '__main__':
  tf.app.run()
