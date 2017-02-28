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

import gflags
flags = gflags
FLAGS = gflags.FLAGS

flags.DEFINE_integer('step', 5, '')

import util
import model 

m = model.Model()
m.batch_size()
for i in xrange(FLAGS.step):
  print(m.batch_size())

def main(argv):
  FLAGS(argv)
  print(FLAGS.name)
  m = model.Model()
  m.batch_size()
  for i in xrange(FLAGS.step):
    print(m.batch_size())
if __name__ == "__main__":
  main(sys.argv)
