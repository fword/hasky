#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   create-model.py
#        \author   chenghuige  
#          \date   2016-10-04 22:31:53.257072
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

sess = tf.InteractiveSession()

with tf.variable_scope('old'):
  w = tf.get_variable('w', shape=[1], initializer=tf.constant_initializer(1.0))

sess.run(tf.initialize_all_variables())

tf.train.Saver().save(sess, '/tmp/old.model')

