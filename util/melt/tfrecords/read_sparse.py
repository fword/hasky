#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   read_sparse.py
#        \author   chenghuige  
#          \date   2016-08-15 20:13:06.751843
#   \Description  @TODO https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/contrib/slim/python/slim/data/
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def read(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return [serialized_example]

 
def inputs(files, decode, batch_size=64,
           num_epochs = None, num_preprocess_threads=12, shuffle=True, 
           batch_join=True, min_after_dequeue = 20000):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read serialized_example
  2. shuffle serialized_examples
  3. decdoe batch_serialized_examples
  notice read_sparse.inputs and also be used for dense inputs,but if you 
  only need to decode part from serialized_example, then read.inputs will 
  be better, less to put to suffle
  #--------decode example, can refer to libsvm-decode.py
  # def decode(batch_serialized_examples):
  #   features = tf.parse_example(
  #       batch_serialized_examples,
  #       features={
  #           'label' : tf.FixedLenFeature([], tf.int64),
  #           'index' : tf.VarLenFeature(tf.int64),
  #           'value' : tf.VarLenFeature(tf.float32),
  #       })

  #   label = features['label']
  #   index = features['index']
  #   value = features['value']

  #   return label, index, value 
  Args:
  decode: user defined decode 
  """
  if not isinstance(files, (list, tuple)):
    files = tf.gfile.Glob(files)
  if not num_epochs: num_epochs = None

  if shuffle == False: batch_join = False

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      files, 
      num_epochs=num_epochs,
      shuffle=shuffle)
    
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    #@TODO cifa10 always use factor = 3, 3 * batch_size, check which is better
    factor = num_preprocess_threads + 1
    if batch_join:
      batch_list = [read(filename_queue) for _ in xrange(num_preprocess_threads)]
      #print batch_list
      batch_serialized_examples = tf.train.shuffle_batch_join(
          batch_list, 
          batch_size=batch_size, 
          capacity=min_after_dequeue + factor * batch_size,
          min_after_dequeue=min_after_dequeue)
    else:
      serialized_example = read(filename_queue)
      if shuffle:	      
        batch_serialized_examples = tf.train.shuffle_batch(	
            serialized_example, 
            batch_size=batch_size, 
            num_threads=num_preprocess_threads,
            capacity=min_after_dequeue + factor * batch_size,
            min_after_dequeue=min_after_dequeue)		    
      else:	    
        batch_serialized_examples = tf.train.batch(
            serialized_example, 
            batch_size=batch_size, 
            num_threads=num_preprocess_threads,
            capacity=min_after_dequeue + factor * batch_size)

    return decode(batch_serialized_examples)

  
