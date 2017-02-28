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
flags.DEFINE_string('vocab', './vocab.bin', 'vocabulary binary file')
flags.DEFINE_string('image_feature_place', 'image_feature:0', '')
flags.DEFINE_string('text_place', 'text:0', '')
flags.DEFINE_string('algo', 'bow', '')
flags.DEFINE_string('model_dir', './models/bow', '')

import sys, os
import melt
import deepiu

import conf
from conf import IMAGE_FEATURE_LEN

from deepiu.image_caption import text2ids

import numpy as np

def parse_keywords(keyword_results):
  keywords = []
  gbdt_scores = []
  for item in keyword_results:
    pos = item.rindex(':')
    keyword = item[:pos]
    keywords.append(keyword)
    score_list = item[pos + 1:].split()
    gbdt_scores.append(score_list[1])
  return keywords, gbdt_scores

def bulk_predict(predictor, images, texts):
  scores = predictor.inference('score', 
                               { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images,
                                 '%s/%s'%(FLAGS.algo, FLAGS.text_place): texts })
  return scores[0]

def predict(predictor):
  for line in sys.stdin:
    l = line.rstrip().split('\t')
    simid = l[0]
    feature = l[1: 1 + IMAGE_FEATURE_LEN]
    feature = np.array([[float(x) for x in feature]])
    keyword_results = l[1 + IMAGE_FEATURE_LEN:]
    keywords, gbdt_scores = parse_keywords(keyword_results)
    ids = text2ids.texts2ids(keywords)
    print(simid)
    scores = bulk_predict(predictor, feature, ids)

    for keyword, gbdt_score, score in zip(keywords, gbdt_scores, scores):
      print(keyword, gbdt_score, score)

def main(_):
  text2ids.init()
  predictor = melt.Predictor(FLAGS.model_dir)
  predict(predictor)

if __name__ == '__main__':
  tf.app.run()