#!/usr/bin/env python
# ==============================================================================
#          \file   input.py
#        \author   chenghuige  
#          \date   2016-08-17 23:50:47.335840
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt

import conf
from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS

def _decode(example, parse, dynamic_batch_length):
  features = parse(
      example,
      features={
          'image_name': tf.FixedLenFeature([], tf.string),
          'image_feature': tf.FixedLenFeature([IMAGE_FEATURE_LEN], tf.float32),
          'text': tf.VarLenFeature(tf.int64),
          'text_str': tf.FixedLenFeature([], tf.string),
      })

  image_name = features['image_name']
  image_feature = features['image_feature']
  text = features['text']
  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  text_str = features['text_str']
  
  return image_name, image_feature, text, text_str

def decode_examples(serialized_examples, dynamic_batch_length):
  return _decode(serialized_examples, tf.parse_example, dynamic_batch_length)

def decode_example(serialized_example, dynamic_batch_length):
  return _decode(serialized_example, tf.parse_single_example, dynamic_batch_length)

#---------------for negative sampling using tfrecords
def _decode_neg(example, parse, dynamic_batch_length):
  features = parse(
      example,
      features={
          'text': tf.VarLenFeature(tf.int64),
          'text_str': tf.FixedLenFeature([], tf.string),
      })

  text = features['text']
  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  text_str = features['text_str']
  
  return text, text_str


def decode_neg_examples(serialized_examples, dynamic_batch_length):
  return _decode_neg(serialized_examples, tf.parse_example, dynamic_batch_length)

def decode_neg_example(serialized_example):
  return _decode_neg(serialized_example, tf.parse_single_example, dynamic_batch_length)

#-----------utils
def get_decodes(shuffle_then_decode, dynamic_batch_length, use_neg=True):
  if shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = lambda x: decode_examples(x, dynamic_batch_length)
    decode_neg = (lambda x: decode_neg_examples(x, dynamic_batch_length)) if use_neg else None
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode = lambda x: decode_example(x, dynamic_batch_length)
    decode_neg = (lambda x: decode_neg_example(x, dynamic_batch_length)) if use_neg else None
  return inputs, decode, decode_neg

def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops
