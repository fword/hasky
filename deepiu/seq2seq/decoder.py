#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decoder.py
#        \author   chenghuige  
#          \date   2016-12-23 23:59:30.933573
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
class Decoder(object):
  def __init__(self):
    pass

  def set_embedding(self, emb):
    self.emb = emb 