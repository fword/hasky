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
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt

import conf
from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS

from deepiu.util import vocabulary 

#flags.DEFINE_boolean('encode_start_mark', False, """need <S> start mark""")
#flags.DEFINE_boolean('encode_end_mark', False, """need </S> end mark""")
#flags.DEFINE_string('encoder_end_mark', '</S>', "or <GO> if NUM_RESERVED_IDS >=3, will use id 2  <PAD2> as <GO>, especailly for seq2seq encoding")

#flags.DEFINE_boolean('add_text_start', True, """if True will add <s> or 0 or GO before text 
#                                              as first input before image, by default will be GO, make add_text_start==True if you use seq2seq""")
#flags.DEFINE_boolean('zero_as_text_start', False, """if add_text_start, 
#                                                    here True will add 0 False will add <s>
#                                                    0 means the loss from image to 0/pad not considered""")
#flags.DEFINE_boolean('go_as_text_start', True, """ """)

#flags.DEFINE_boolean('input_with_start_mark', False, """if input has already with <S> start mark""")
#flags.DEFINE_boolean('input_with_end_mark', False, """if input has already with </S> end mark""")

flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

encoder_end_id = None
def get_decoder_start_id():
  #start_id = vocabulary.start_id()
  start_id = None
  if not FLAGS.input_with_start_mark and FLAGS.add_text_start:
    if FLAGS.zero_as_text_start:
      start_id = 0
    elif FLAGS.go_as_text_start:
      start_id = vocabulary.go_id()
    else:
      start_id = vocabulary.start_id()
  return start_id

def get_decoder_end_id():
  if (FLAGS.input_with_end_mark):
    return None 
  else:
    return vocabulary.end_id()


def _decode(example, parse, dynamic_batch_length):
  features = parse(
      example,
      features={
          'image_name': tf.FixedLenFeature([], tf.string),
          'url': tf.FixedLenFeature([], tf.string),
          'text_str': tf.FixedLenFeature([], tf.string),
          'ct0_str': tf.FixedLenFeature([], tf.string),
          'title_str': tf.FixedLenFeature([], tf.string),
          'real_title_str': tf.FixedLenFeature([], tf.string),
          'text': tf.VarLenFeature(tf.int64),
          'ct0': tf.VarLenFeature(tf.int64),
          'title': tf.VarLenFeature(tf.int64),
          'real_title': tf.VarLenFeature(tf.int64),
      })

  image_name = features['image_name']
  text = features['text']
  input_type = 'real_title'
  input_text = features[input_type]

  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)

  text, _ = melt.pad(text, start_id= get_decoder_start_id(), end_id=get_decoder_end_id())
  
  input_maxlen = 0 if dynamic_batch_length else INPUT_TEXT_MAX_WORDS
  input_text = melt.sparse_tensor_to_dense(input_text, maxlen)
  
  input_text, _ = melt.pad(input_text, 
                           start_id=(vocabulary.vocab.start_id() if FLAGS.encode_start_mark else None),
                           end_id=(encoder_end_id if FLAGS.encode_end_mark else None))


  text_str = features['text_str']
  input_text_str = features['{}_str'.format(input_type)]
  
  return image_name, text, text_str, input_text, input_text_str

def decode_examples(serialized_examples, dynamic_batch_length):
  return _decode(serialized_examples, tf.parse_example, dynamic_batch_length)

def decode_example(serialized_example, dynamic_batch_length):
  return _decode(serialized_example, tf.parse_single_example, dynamic_batch_length)


#-----------utils
def get_decodes(shuffle_then_decode, dynamic_batch_length):
  vocabulary.init()

  global encoder_end_id
  if FLAGS.encoder_end_mark == '</S>':
    encoder_end_id =  vocabulary.end_id()
  else:
    encoder_end_id = vocabulary.go_id() #NOTICE NUM_RESERVED_IDS must >= 3 TODO
  assert encoder_end_id != vocabulary.vocab.unk_id(), 'input vocab generated without end id'

  if shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = lambda x: decode_examples(x, dynamic_batch_length)
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode = lambda x: decode_example(x, dynamic_batch_length)
  return inputs, decode

