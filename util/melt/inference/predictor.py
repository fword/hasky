#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predictor.py
#        \author   chenghuige  
#          \date   2016-09-30 15:19:35.984852
#   \Description  
# ==============================================================================
"""
This predictor will read from checkpoint and meta graph, only depend on tensorflow
no other code dpendences, so can help hadoop or online deploy,
and also can run inference without code of building net/graph
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import os, sys

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path):
  #  raise ValueError(model_path)
  return os.path.dirname(model_path), model_path

class Predictor(object):
  def __init__(self, model_dir=None, meta_graph=None, model_name=None):
    super(Predictor, self).__init__()
    self.sess = tf.InteractiveSession()
    #ops will be map and internal list like
    #{ text : [op1, op2], text2 : [op3, op4, op5], text3 : [op6] }
    if model_dir is not None:
      self.restore(model_dir, meta_graph, model_name)

  def inference(self, key, feed_dict=None, index=0):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(tf.get_collection(key)[index], feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      keys = [tf.get_collection(key)[index] for key,index in zip(keys, indexes)]
      return self.sess.run(keys, feed_dict=feed_dict)

  def predict(self, key, feed_dict=None, index=0):
    return self.inference(key, feed_dict, index)

  def restore(self, model_dir, meta_graph=None, model_name=None, random_seed=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    model_dir, model_path = get_model_dir_and_path(model_dir, model_name)
    self.model_path = model_path
    if meta_graph is None:
      meta_graph = '%s.meta'%model_path
    print('restore from %s'%model_path, file=sys.stderr)
    saver = tf.train.import_meta_graph(meta_graph)
    print('import graph ok %s'%meta_graph, file=sys.stderr)
    saver.restore(self.sess, model_path)
    print('restore ok %s'%model_path, file=sys.stderr)
    if random_seed is not None:
      tf.set_random_seed(random_seed)
    return self.sess