#!/usr/bin/env python
# ==============================================================================
#          \file   predictor_base.py
#        \author   chenghuige  
#          \date   2016-08-17 23:57:11.987515
#   \Description  
# ==============================================================================

"""
This is used for train predict, predictor building graph not read from meta
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt

def get_tensor_from_key(key, index=-1):
  if isinstance(key, str):
    try:
      return tf.get_collection(key)[index]
    except Exception:
      print('Warning:', key, ' not find in graph')
      return tf.no_op()
  else:
    return key

class PredictorBase(object):
  def __init__(self, sess=None):
    super(PredictorBase, self).__init__()
    if sess is None:
      self.sess = melt.get_session()
    else:
      self.sess = sess

  def load(self, model_dir, var_list=None, model_name=None, sess = None):
    """
    only load varaibels from checkpoint file, you need to 
    create the graph before calling load
    """
    if sess is not None:
      self.sess = sess
    self.model_path = melt.get_model_path(model_dir, model_name)
    saver = melt.restore_from_path(self.sess, self.model_path, var_list)
    return self.sess

  def restore_from_graph(self):
    pass

  def restore(self, model_dir, model_name=None, sess=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    if sess is not None:
      self.sess = sess
    self.model_path = model_path = melt.get_model_path(model_dir, model_name)
    meta_filename = '%s.meta'%model_path
    saver = tf.train.import_meta_graph(meta_filename)
    self.restore_from_graph()
    saver.restore(self.sess, model_path)
    return self.sess

  def run(self, key, feed_dict=None):
    return self.sess.run(key, feed_dict)

  def inference(self, key, feed_dict=None, index=-1):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(get_tensor_from_key(key, index), feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      keys = [get_tensor_from_key(key, index) for key,index in zip(keys, indexes)]
      return self.sess.run(keys, feed_dict=feed_dict)