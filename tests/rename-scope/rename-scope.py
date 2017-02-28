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

saver = tf.train.import_meta_graph('/tmp/old.model.meta')

saver.restore(sess, '/tmp/old.model')

src_vars = [v for v in tf.all_variables() if v.name.startswith('old')]
print('old_vars:', [v.name for v in src_vars])

out_vars = {v.name[:v.name.rfind(':')].replace('old', 'new', 1): v for v in src_vars}
print('new_vars:', [key for key in out_vars])

tf.train.Saver(var_list=out_vars).save(sess, '/tmp/new.model')

