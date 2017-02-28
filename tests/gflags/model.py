#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2016-08-13 20:28:06.145474
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import tensorflow as tf

import gflags
flags = gflags
FLAGS = gflags.FLAGS
flags.DEFINE_integer('batch_size', 100, '')

class Model():
  def __init__(self):
    self.batch_size_ = FLAGS.batch_size 

  def batch_size(self):
    return self.batch_size_
