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

flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS 

flags.DEFINE_integer('batch_size', 100, '')

my_batch = None

def init():
  global my_batch
  if my_batch is None:
    my_batch = FLAGS.batch_size 

def batch_size():
  print(FLAGS.name)
  return FLAGS.batch_size
