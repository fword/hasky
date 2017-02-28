#!/usr/bin/env python
# ==============================================================================
#          \file   train-melt.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import melt
train_flow = melt.flow.train_tfrecord.simple_train_flow
inputs = melt.read_sparse.inputs
decode = melt.libsvm_decode.decode
from melt.models import Mlp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 34 
NUM_FEATURES = 324510

def build_graph(X, y):
  algo = Mlp(NUM_FEATURES, NUM_CLASSES)
  py_x = algo.forward(X)
  
  #loss = melt.sparse_softmax_cross_entropy(py_x, y)
  loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, y))

  accuracy = melt.precision_at_k(py_x, y, 1)

  return loss, accuracy


def train():
  trainset = sys.argv[1]
  X, y = inputs(
    trainset, 
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_preprocess_threads=FLAGS.num_preprocess_threads,
    batch_join=FLAGS.batch_join,
    shuffle=FLAGS.shuffle)
  
  train_with_validation = len(sys.argv) > 2
  if train_with_validation:
    validset = sys.argv[2]
    eval_X, eval_y = inputs(
      validset, 
      decode=decode,
      batch_size=FLAGS.batch_size * 10,
      num_preprocess_threads=FLAGS.num_preprocess_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)

  loss, accuracy = build_graph(X, y)
  train_op = melt.gen_train_op(loss, FLAGS.learning_rate)
  if train_with_validation:
    tf.get_variable_scope().reuse_variables()
    _, eval_accuracy = build_graph(eval_X, eval_y)
    eval_ops = [eval_accuracy]
  else:
    eval_ops = None

  train_flow([train_op, loss, accuracy], 
             deal_results=melt.show_precision_at_k,
             eval_ops=eval_ops,
             deal_eval_results=None,
             print_avg_loss=True
             )

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
