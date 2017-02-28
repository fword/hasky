#!/usr/bin/env python
# ==============================================================================
#          \file   test-melt.py
#        \author   chenghuige  
#          \date   2016-08-17 15:34:45.456980
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import melt
test_flow = melt.flow.test_tfrecord.test_flow
inputs = melt.read_sparse.inputs
decode = melt.libsvm_decode.decode
from melt.models import Mlp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')

flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_string('model_path', './model', '')

flags.DEFINE_integer('eval_times', 0, '')
flags.DEFINE_integer('num_steps', 100, '')

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 34 
NUM_FEATURES = 324510

def build_graph(X, y):
  algo = Mlp(NUM_FEATURES, NUM_CLASSES)
  py_x = algo.forward(X)
  
  loss = melt.sparse_softmax_cross_entropy(py_x, y)
  #loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, y))

  accuracy = melt.precision_at_k(py_x, y, 1)

  return loss, accuracy


def test():
  X, y = inputs(
    sys.argv[1], 
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=0,  #for test we will set num_epochs as 1! but now has epoch variable model loading problem, so set 0..
    num_preprocess_threads=FLAGS.num_preprocess_threads,
    batch_join=FLAGS.batch_join,
    shuffle=FLAGS.shuffle)

  loss, accuracy = build_graph(X, y)
   
  eval_names = ['loss', 'precision@1']
  print('eval_names:', eval_names)
  
  test_flow(
    [loss, accuracy], 
    names=eval_names,
    model_dir=FLAGS.model_path, 
    num_steps=FLAGS.num_steps,
    eval_times=FLAGS.eval_times)

def main(_):
  test()

if __name__ == '__main__':
  tf.app.run()

  
