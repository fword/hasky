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
train_flow = melt.flow.train_tfrecord.train_flow

from melt.models import Mlp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
#-----------@TODO so setting num_epochs non 0, will casue load fail, Attempting to use uninitialized value input/input_producer/limit_epochs/epochs 
#looks like tf examples all train using num_epochs as None,so right now train just use stpes not set num_epochs and for test can set num_epochs
flags.DEFINE_integer('num_epochs', 0, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('eval_interval_steps', 100, '')

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 34 
NUM_FEATURES = 324510

def build_graph(X, y):
  algo = Mlp(input_dim=NUM_FEATURES, num_classes=NUM_CLASSES)
  py_x = algo.forward(X)
  
  loss = melt.sparse_softmax_cross_entropy(py_x, y)
  tf.scalar_summary('loss_%s'%loss.name, loss)
 
  #tf.scalar_summary('loss', loss)
  ##loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, y))

  accuracy = melt.precision_at_k(py_x, y, 1)
  tf.scalar_summary('precision@1_%s'%accuracy.name, accuracy)

  #below will cause  tensorflow.python.framework.errors.InvalidArgumentError: Duplicate tag precicsion@1 found in summary inputs
  #the problem here is we want to share all other things but without scalar summarys in graph
  #so if we want to build more than once ... then scalar_summary must use op.name
  #or else do it outof build_graph, setting names by yourself!
  #eval_loss, eval_accuracy = build_grapy(X, y)
  #tf.scalar_summary('eval_loss', eval_loss) 
  #tf.scalar_summary('eval_accuracy', eval_accuracy)
  #since tensorboard has 'Split on underscores', so for better comparaion side by side
  #loss_train, loss_eval, accuracy_train, accuracy_eval is better then train_loss,eval_loss 

  #tf.scalar_summary('precicsion@1', accuracy)
  return loss, accuracy

def train():
  trainset = sys.argv[1]
  inputs = melt.read_sparse.inputs
  decode = melt.libsvm_decode.decode
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
    eval_loss, eval_accuracy = build_graph(eval_X, eval_y)
    tf.scalar_summary('loss_eval', eval_loss)
    eval_ops = [eval_loss, eval_accuracy]
  else:
    eval_ops = None

  train_flow([train_op, loss, accuracy], 
             deal_results=melt.show_precision_at_k,
             #deal_results=None,
             eval_ops=eval_ops,
             deal_eval_results= lambda x: melt.print_results(x, names=['precision@1']),
             print_avg_loss=True,
             eval_interval_steps=FLAGS.eval_interval_steps
             )

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
