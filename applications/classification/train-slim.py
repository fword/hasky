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
import model_slim as model

import decode 
decode = decode.decode_examples

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
#-----------@TODO so setting num_epochs non 0, will casue load fail, Attempting to use uninitialized value input/input_producer/limit_epochs/epochs 
#looks like tf examples all train using num_epochs as None,so right now train just use stpes not set num_epochs and for test can set num_epochs
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('eval_interval_steps', 200, '')

#@FIXME seems ~/data/..  not work.. why using sys.argv[1] ok ?
flags.DEFINE_string('train_files_pattern', '/home/gezi/data/urate/tfrecord/train', 'train files pattern')
flags.DEFINE_string('valid_files_pattern', '/home/gezi/data/urate/tfrecord/test', """
                                                      valid/test files pattern during training
                                                      setting to empty means no validation
                                                      """)

def train():
  trainset = FLAGS.train_files_pattern
  trainset = sys.argv[1]
  print('trainset', trainset)
  inputs = melt.shuffle_then_decode.inputs
  X, y = inputs(
    trainset, 
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_preprocess_threads=FLAGS.num_preprocess_threads,
    batch_join=FLAGS.batch_join,
    shuffle=FLAGS.shuffle)
  
  train_with_validation = bool(FLAGS.valid_files_pattern)
  if train_with_validation:
    validset = FLAGS.valid_files_pattern
    eval_X, eval_y = inputs(
      validset, 
      decode=decode,
      batch_size=FLAGS.batch_size * 10,
      num_preprocess_threads=FLAGS.num_preprocess_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)
  
  loss, accuracy = model.build_graph(X, y)
  train_op = melt.gen_train_op(loss, FLAGS.learning_rate)
  if train_with_validation:
    tf.get_variable_scope().reuse_variables()
    eval_loss, eval_accuracy = model.build_graph(eval_X, eval_y)
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
