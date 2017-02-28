#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description   @TODO why not shuffle will be much slower then shuffle..
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
python ./read-records-melt.py input
'''
import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_boolean('shuffle_then_decode', True, '')

def read_once(sess, step, ops):
  id, X, y = sess.run(ops)
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()
  if step % 10000 == 0:
    print('duration:', read_once.timer.elapsed())
    if step == 0:
      print(id)
      print(X)
      print(y)

num_features = 488
def decode_examples(batch_serialized_examples):
  features = tf.parse_example(
    batch_serialized_examples,
    features={
        'id' : tf.FixedLenFeature([], tf.int64),
        'label' : tf.FixedLenFeature([], tf.int64),
        'feature' : tf.FixedLenFeature([num_features], tf.float32),
    })

  id = features['id']
  label = features['label']
  feature = features['feature']

  return id, feature, label

#well need 488 set for feature so this is an diadvantage, though we enocde example feature as fixedlen, 
#when decode we still need to set the length shape.. @TODO can we remove 488?
#TypeError: DataType int32 for attr 'Tdense' not in list of allowed values: float32, int64, string

def decode_example(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
    features={
        'id' : tf.FixedLenFeature([], tf.int64),
        'label' : tf.FixedLenFeature([], tf.int64),
        'feature' : tf.FixedLenFeature([num_features], tf.float32),
    })

  id = features['id']
  label = features['label']
  feature = features['feature']

  return id, feature, label

from melt.flow import tf_flow
def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  if FLAGS.shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = decode_examples
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode = decode_example

  #looks like setting sparse==1 or 0 all ok, but sparse=1 is faster...
  #may be for large example and you only decode a small part features then sparse==0 will be
  #faster since decode before shuffle, shuffle less data
  #but sparse==0 for one flow can deal both sparse input and dense input
  
  with tf.Graph().as_default():
    id, X, y = inputs(
      sys.argv[1], 
      decode=decode,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)
    
    tf_flow(lambda sess, step: read_once(sess, step, [id, X, y]))
    

def main(_):
  read_records()


if __name__ == '__main__':
  tf.app.run()
