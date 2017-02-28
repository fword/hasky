#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2016-08-12 11:52:01.952044
#   \Description   gen records for melt dense format data
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import tensorflow as tf
import numpy as np

import melt

def main(argv):
  writer = tf.python_io.TFRecordWriter(argv[2])
  num = 0
  for line in open(argv[1]):
    if line[0] == '#':
      continue
    if num % 10000 == 0:
      print('%d lines done'%num)
    l = line.rstrip().split()
    
    label_index = 0
    if l[0][0] == '_':
      label_index = 1
      id = int(l[0][1:])
    else:
      id = num
    label = int(l[label_index])
    
    start = label_index + 1
    #notice this will be float64 not float32
    feature = np.array([float(x) for x in l[start:]])
    if num == 0:
      print('len feature', len(feature))
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': melt.int_feature(id), 
        'label': melt.int_feature(label),
        'feature': melt.bytes_feature(feature.tostring()),
        'length': melt.int_feature(len(feature)),
        }))
    writer.write(example.SerializeToString())
    num += 1

if __name__ == '__main__':
  tf.app.run()
