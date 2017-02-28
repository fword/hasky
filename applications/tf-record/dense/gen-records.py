#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2016-08-12 11:52:01.952044
#   \Description   gen records for melt dense format data
# ==============================================================================

  
"""
python gen-records.py /home/gezi/data/urate/train /tmp/urate.train --num_examples 100
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_examples', 0, 'Batch size.')

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
    feature = [float(x) for x in l[start:]]
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
        'id': melt.int_feature(id), 
        'label': melt.int_feature(label),
        'feature': melt.float_feature(feature),
        }))
    writer.write(example.SerializeToString())
    num += 1
    if FLAGS.num_examples and num == FLAGS.num_examples:
      break

if __name__ == '__main__':
  tf.app.run()
