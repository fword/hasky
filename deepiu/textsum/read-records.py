#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import tensorflow as tf
from collections import defaultdict

import numpy as np

from gezi import Timer
import melt 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')
flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_string('input', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/fixed_valid/test', '')
flags.DEFINE_string('name', 'train', 'records name')
flags.DEFINE_boolean('dynamic_batch_length', True, '')
flags.DEFINE_boolean('shuffle_then_decode', True, '')

max_index = 0
def read_once(sess, step, ops):
  global max_index
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()

  image_name, text, text_str, input_text, input_text_str = sess.run(ops)
  
  if step % 100 == 0:
    print('step:', step)
    print('duration:', read_once.timer.elapsed())
    print('image_name:', image_name[0])
    
    print('text:', text[0])
    print('text_str:', text_str[0])
    print('len(text_str):', len(text_str[0]))

    print('input_text:', input_text[0])
    print('input_text_str:', input_text_str[0])
    print('len(input_text_str):', len(input_text_str[0]))

  cur_max_index = np.max(text)
  if cur_max_index > max_index:
    max_index = cur_max_index


from melt.flow import tf_flow
import input
def read_records():
  inputs, decode = input.get_decodes(FLAGS.shuffle_then_decode, FLAGS.dynamic_batch_length)

  ops = inputs(
    FLAGS.input,
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_threads,
    #num_threads=1,
    batch_join=FLAGS.batch_join,
    shuffle_batch=FLAGS.shuffle_batch,
    shuffle=FLAGS.shuffle,
    #fix_random=True,
    #fix_sequence=True,
    #no_random=True,
    allow_smaller_final_batch=True,
    )
  print(ops) 
  
  timer = Timer()
  tf_flow(lambda sess, step: read_once(sess, step, ops))
  print('max_index:', max_index)
  print(timer.elapsed())
    

def main(_):
  read_records()

if __name__ == '__main__':
  tf.app.run()
