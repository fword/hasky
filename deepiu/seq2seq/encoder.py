#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder.py
#        \author   chenghuige  
#          \date   2016-12-23 23:59:26.165659
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class Encoder(object):
  def __init__(self):
    pass

  def set_embedding(self, emb):
    self.emb = emb 