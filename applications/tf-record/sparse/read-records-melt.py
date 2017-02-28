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

import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 

import functools

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_string('label_type', 'int', '')

def read_once(sess, step, ops):
  X, y = sess.run(ops)
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()
  if step % 10000 == 0:
    print('duration:', read_once.timer.elapsed())
    if step == 0:
      print(X)
      print(y)

from melt.flow import tf_flow
def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  inputs = melt.read_sparse.inputs
  label_type = tf.int32 if FLAGS.label_type == 'int' else tf.float32
  decode = functools.partial(melt.libsvm_decode.decode, label_type=label_type)

  with tf.Graph().as_default():
    X, y = inputs(
      sys.argv[1], 
      decode=decode,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_preprocess_threads=FLAGS.num_preprocess_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)
    
    tf_flow(lambda sess, step: read_once(sess, step, [X, y]))
    

def main(_):
  read_records()


if __name__ == '__main__':
  tf.app.run()
