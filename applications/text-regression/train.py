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
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_string('model_dir', '/home/gezi/temp/text-classification', '')

#--------- modify below according to your data!
flags.DEFINE_integer('num_features', -1, '')

import gezi
import melt
logging = melt.logging

import model
import functools
decode = functools.partial(melt.libsvm_decode.decode, label_type=tf.float32)

def gen_predict_graph():
  predictor = model.Predictor()
  score = model.predict(predictor.X)
  tf.add_to_collection('score', score)

def train():
  assert FLAGS.num_features > 0, 'you must pass num_features according to your data'
  print('num_features:', FLAGS.num_features)
  model.set_input_info(num_features=FLAGS.num_features)

  trainset = sys.argv[1]
  inputs = melt.shuffle_then_decode.inputs
  X, y = inputs(
    trainset, 
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_preprocess_threads,
    batch_join=FLAGS.batch_join,
    shuffle=FLAGS.shuffle)
  
  train_with_validation = len(sys.argv) > 2
  if train_with_validation:
    validset = sys.argv[2]
    eval_X, eval_y = inputs(
      validset, 
      decode=decode,
      batch_size=FLAGS.batch_size * 10,
      num_threads=FLAGS.num_preprocess_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle)
  
  with tf.variable_scope('main') as scope:
    loss, accuracy = model.build_graph(X, y)
    scope.reuse_variables()
    gen_predict_graph()
    if train_with_validation:
      eval_loss, eval_accuracy = model.build_graph(eval_X, eval_y)
      eval_ops = [eval_loss, eval_accuracy]
    else:
      eval_ops = None

  melt.apps.train_flow(
             [loss, accuracy], 
             deal_results=melt.show_precision_at_k,
             eval_ops=eval_ops,
             deal_eval_results= lambda x: melt.print_results(x, names=['precision@1']),
             model_dir=FLAGS.model_dir)

def main(_):
  logging.set_logging_path(gezi.get_dir(FLAGS.model_dir))
  train()

if __name__ == '__main__':
  tf.app.run()
