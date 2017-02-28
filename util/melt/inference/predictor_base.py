#!/usr/bin/env python
# ==============================================================================
#          \file   predictor_base.py
#        \author   chenghuige  
#          \date   2016-08-17 23:57:11.987515
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt

class PredictorBase(object):
  def __init__(self):
    super(PredictorBase, self).__init__()
    self.sess = melt.get_session()

  def load(self, model_dir, var_list=None, model_name=None):
    """
    only load varaibels from checkpoint file, you need to 
    create the graph before calling load
    """
    self.model_path = melt.get_model_path(model_dir, model_name)
    saver = melt.restore_from_path(self.sess, self.model_path, var_list)
    return self.sess

  def restore_from_graph(self):
    pass

  def restore(self, model_dir, model_name=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    self.model_path = model_path = melt.get_model_path(model_dir, model_name)
    meta_filename = '%s.meta'%model_path
    saver = tf.train.import_meta_graph(meta_filename)
    self.restore_from_graph()
    saver.restore(self.sess, model_path)
    return self.sess
