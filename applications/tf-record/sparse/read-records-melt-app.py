#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description   @depreciated, this one not work any more, melt.apps depreciated
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 
inputs = melt.apps.read.sparse_inputs
decode = melt.libsvm_decode.decode

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')

def read_once(sess, step, ops):
  X, y = sess.run(ops)
  if not hasattr(read_once, 'timer'):
    read_once.timer = Timer()
  if step % 10000 == 0:
    print('duration:', read_once.timer.elapsed())
    if step == 0:
      print(X)
      print(y)
      print(y.shape)

from melt.flow import tf_flow
def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    X, y = inputs(sys.argv[1], decode=decode, batch_size=FLAGS.batch_size)
    
    tf_flow(lambda sess, step: read_once(sess, step, [X, y]))
    

def main(_):
  read_records()


if __name__ == '__main__':
  tf.app.run()
