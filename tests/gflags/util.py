#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-20 23:42:06.016040
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gflags
flags = gflags
FLAGS = gflags.FLAGS

flags.DEFINE_string('name', 'messi', '')

print(FLAGS.name)
