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

flags.DEFINE_string('model_dir', '/home/gezi/temp/text-regression', '')

import sys, os
import melt
import numpy as np

class SparseFeatures(object):
  def __init__(self):
    self.sp_indices = [] 
    self.start_indices = [0]
    self.sp_ids_val = [] 
    self.sp_weights_val = []
    self.sp_shape = None

  def mini_batch(self, start, end):
    batch = SparseFeatures()
    start_ = self.start_indices[start]
    end_ = self.start_indices[end]
    batch.sp_ids_val = self.sp_ids_val[start_: end_]
    batch.sp_weights_val = self.sp_weights_val[start_: end_]
    row_idx = 0
    max_len = 0
    #@TODO better way to construct sp_indices for each mini batch ?
    for i in xrange(start + 1, end + 1):
      len_ = self.start_indices[i] - self.start_indices[i - 1]
      if len_ > max_len:
        max_len = len_
      for j in xrange(len_):
        batch.sp_indices.append([i - start - 1, j])
      row_idx += 1 
    batch.sp_shape = [end - start, max_len]
    return batch

  def full_batch(self):
    if len(self.sp_indices) == 0:
      row_idx = 0
      max_len = 0
      for i in xrange(1, len(self.start_indices)):
        len_ = self.start_indices[i] - self.start_indices[i - 1]
        if len_ > max_len:
          max_len = len_
        for j in xrange(len_):
          self.sp_indices.append([i - 1, j])
        row_idx += 1 
    self.sp_shape = [len(self.start_indices) - 1, max_len]


def bulk_predict(predictor, features):
  features.full_batch()
  scores = predictor.inference('score', 
                               feed_dict={
                                 'main/indices:0': features.sp_indices,
                                 'main/weights_val:0': features.sp_weights_val,
                                 'main/ids_val:0': features.sp_ids_val})
  return scores

#libsvm format input
def predict(predictor, buffer_size=100):
  labels = []
  start_idx = 0
  features = SparseFeatures()
  for line in sys.stdin:
    l = line.rstrip().split()

    labels.append(float(l[0]))
    start_idx += len(l) - 1
    features.start_indices.append(start_idx)
    for item in l[1:]:
      id, val = item.split(':')
      features.sp_ids_val.append(int(id))
      features.sp_weights_val.append(float(val))

    if len(labels) == buffer_size:
      scores = bulk_predict(predictor, features)
      for label, score in zip(labels, scores):
        print(label, score)

      labels = []
      start_idx = 0
      features = SparseFeatures()

  if len(labels) > 0:
    scores = bulk_predict(predictor, features)
    for label, score in zip(labels, scores):
      print(label, score)

def main(_):
  predictor = melt.Predictor(FLAGS.model_dir)
  predict(predictor)

if __name__ == '__main__':
  tf.app.run()