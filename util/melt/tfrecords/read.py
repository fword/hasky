#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2016-08-15 20:10:44.183328
#   \Description   Read from TFRecords
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#change from 1000 to 500, which will be a bit safer in muliti thread env reading hdfs using libhdfs TODO check
MIN_AFTER_QUEUE = 500

#----------------below  deprecated
def read_decode(filename_queue, decode):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  values = decode(serialized_example)
  #---for safe, or decode can make sure this for single value turn to list []
  if not isinstance(values, (list, tuple)):
    values = [values]
  return values

def inputs(files, decode, batch_size=64,
           num_epochs = None, num_preprocess_threads=12, shuffle=True, 
           batch_join=True, min_after_dequeue = 20000):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read decode serialized_example
  2. shuffle decoded values
  3. return batch decoded values
  Args:
  decode: user defined decode 
  #---decode example
  # features = tf.parse_single_example(
  #     serialized_example,
  #     features={
  #         'feature': tf.FixedLenFeature([], tf.string),
  #         'name': tf.FixedLenFeature([], tf.string),
  #         'comment_str': tf.FixedLenFeature([], tf.string),
  #         'comment': tf.FixedLenFeature([], tf.string),
  #         'num_words': tf.FixedLenFeature([], tf.int64),
  #     })
  # feature = tf.decode_raw(features['feature'], tf.float32)
  # feature.set_shape([IMAGE_FEATURE_LEN])
  # comment = tf.decode_raw(features['comment'], tf.int64)
  # comment.set_shape([COMMENT_MAX_WORDS])
  # name = features['name']
  # comment_str = features['comment_str']
  # num_words = features['num_words']
  # return name, feature, comment_str, comment, num_words
  Returns:
  list of tensors
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
      batch_list = [read_decode(filename_queue, decode) for _ in xrange(num_preprocess_threads)]
      #print batch_list
      batch = tf.train.shuffle_batch_join(
          batch_list, 
          batch_size=batch_size, 
          capacity=min_after_dequeue + factor * batch_size,
          min_after_dequeue=min_after_dequeue)
    else:
      serialized_example = read_decode(filename_queue, decode)
      if shuffle:	    
          batch = tf.train.shuffle_batch(	
             serialized_example,
             batch_size=batch_size, 
             num_threads=num_preprocess_threads,
             capacity=min_after_dequeue + factor * batch_size,
             min_after_dequeue=min_after_dequeue)
      else:
          batch = tf.train.batch(
             serialized_example, 
             batch_size=batch_size, 
             num_threads=num_preprocess_threads,
             capacity=min_after_dequeue + factor * batch_size)

    return batch
