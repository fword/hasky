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

with tf.variable_scope('new'):
  w = tf.get_variable('w', shape=[1], initializer=tf.constant_initializer(2.0))

tf.train.Saver().restore(sess, '/tmp/new.model')

print('tf.all_variables:', [v.name for v in tf.all_variables()])

print('w val:',  w.eval())
