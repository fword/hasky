#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-13 20:28:13.751362
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import tensorflow as tf

flags = tf.app.flags 
FLAGS = tf.app.flags.FLAGS 

flags.DEFINE_integer('step', 5, '')

#------if using FLAGS.step now
# Traceback (most recent call last):
#   File "train.py", line 28, in <module>
#     model.init()
#   File "/home/gezi/mine/tensorflow-exp/tests/flags/model.py", line 28, in init
#     my_batch = FLAGS.batch_size 
#   File "/usr/lib/python2.7/site-packages/tensorflow/python/platform/flags.py", line 43, in __getattr__
#     raise AttributeError(name)
#print(FLAGS.step)

import util
import model 

model.init()
util.init()

print('util.name:', util.my_name)

def main(argv):
  print(FLAGS.name)
  print(FLAGS.batch_size)
  print(model.batch_size())

  

if __name__ == "__main__":
  tf.app.run()