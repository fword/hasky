#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-14 13:47:05.912129
#   \Description  train mlp classifier for tlc/libsvm format input text classfication
#                 different from train.py here will evaluate during train
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import tensorflow as tf 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

MIN_AFTER_DEQUEUE = 10000

def read(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return serialized_example

def decode(batch_serialized_examples):
  features = tf.parse_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], tf.int64),
          'index' : tf.VarLenFeature(tf.int64),
          'value' : tf.VarLenFeature(tf.float32),
      })

  label = features['label']
  index = features['index']
  value = features['value']

  return label, index, value

def batch_inputs(files, batch_size, num_epochs = None, num_preprocess_threads=1):
  """Reads input data num_epochs times.
  """
  if not num_epochs: num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs)

    serialized_example = read(filename_queue)
    batch_serialized_examples = tf.train.shuffle_batch(
        [serialized_example], 
        batch_size=batch_size, 
        num_threads=num_preprocess_threads,
        capacity=MIN_AFTER_DEQUEUE + (num_preprocess_threads + 1) * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=MIN_AFTER_DEQUEUE)

    return decode(batch_serialized_examples)

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 34 
NUM_FEATURES = 324510

def matmul(X, w):
    """ General matmul  that will deal both for dense and sparse input
    hide the differnce of dense adn sparse input for end users
    """
    if type(X) == tf.Tensor:
        return tf.matmul(X,w)
    else:
        return tf.nn.embedding_lookup_sparse(w, X[0], X[1], combiner="sum")

def get_weights(name, shape, stddev=0.01):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=stddev))

def get_bias(name, shape, val=0.1):
  return tf.get_variable(name, shape, initializer=tf.constant_initializer(val))

class Mlp(object):
  def model(self, X, w_h, b_h, w_o, b_o):
    h = tf.nn.relu(matmul(X, w_h) + b_h) 
    return tf.matmul(h, w_o) + b_o 
  
  def forward(self, X):
    hidden_size = 200
    num_features = NUM_FEATURES
    num_classes = NUM_CLASSES

    with tf.device('/cpu:0'):
      w_h = get_weights('w_h', [num_features, hidden_size]) 
  
    b_h = get_bias('b_h', [hidden_size])
    w_o = get_weights('w_o', [hidden_size, num_classes])
    b_o = get_bias('b_o', [num_classes])
    py_x = self.model(X, w_h, b_h, w_o, b_o)

    return py_x  

def build_graph(X, y):
  algo = Mlp()
  py_x = algo.forward(X)
  
  cost = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, y))

  correct_prediction = tf.nn.in_top_k(py_x, y, 1)

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return cost, accuracy

def gen_optimizer(cost, learning_rate):
  train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)  
  return train_op

def train():
  with tf.Graph().as_default():
    tf_record_pattern = sys.argv[1]
    data_files = tf.gfile.Glob(tf_record_pattern)
    label, index, value = batch_inputs(data_files, 
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_preprocess_threads=FLAGS.num_preprocess_threads)
    
    tf_record_pattern_test = sys.argv[2]
    data_files_test = tf.gfile.Glob(tf_record_pattern_test)
    label_test, index_test, value_test = batch_inputs(data_files_test, 
      batch_size=FLAGS.batch_size * 10,
      num_epochs=None, 
      num_preprocess_threads=FLAGS.num_preprocess_threads)

    X = (index, value)
    cost, accuracy = build_graph(X, label)
    train_op = gen_optimizer(cost, FLAGS.learning_rate)
    
    tf.get_variable_scope().reuse_variables()
    cost_test, accuracy_test = build_graph((index_test, value_test), label_test)
    
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    sess = tf.InteractiveSession()
    sess.run(init_op)

    summary_writer = tf.train.SummaryWriter('/tmp/model', sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        _, cost_, accuracy_ = sess.run([train_op, cost, accuracy]) 
        if step % 100 == 0:
        	print('step:', step, 'train precision@1:', accuracy_,'cost:', cost_)
        	accuracy_test_ = sess.run([accuracy_test])
        	print('test  precision@1:', accuracy_test_)
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
