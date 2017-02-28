#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-renamed.py
#        \author   chenghuige  
#          \date   2016-10-04 22:32:07.116676
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph('/tmp/new.model.meta')

saver.restore(sess, '/tmp/new.model')

print('tf.all_variables:', [v.name for v in tf.all_variables()])
