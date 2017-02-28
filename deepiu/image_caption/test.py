#!/usr/bin/env python
# ==============================================================================
#          \file   test-melt.py
#        \author   chenghuige  
#          \date   2016-08-17 15:34:45.456980
#   \Description  
# ==============================================================================
"""
train_input from input_app now te test_input 
valid_input set empty
python ./test.py --train_input test_input --valid_input ''
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'show_and_tell', 'bow rnn show_and_tell')
flags.DEFINE_string('model_dir', './model.flickr.show_and_tell2/', '')  
flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/seq-with-unk/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_integer('num_interval_steps', 100, '')
flags.DEFINE_integer('eval_times', 0, '')

flags.DEFINE_integer('monitor_level', 2, '1 will monitor emb, 2 will monitor gradient')


import sys

import melt
test_flow = melt.flow.test_flow
import input_app as InputApp
logging = melt.utils.logging

#algos
import algos.algos_factory
from algos.algos_factory import Algos

def test():
  trainer = algos.algos_factory.gen_tranier(FLAGS.algo)
  input_app = InputApp.InputApp()
  sess = input_app.sess = tf.InteractiveSession()

  input_results = input_app.gen_input()
  
  #--------train
  image_name, image_feature, text, text_str = input_results[input_app.input_train_name]
  #--------train neg
  neg_text, neg_text_str = input_results[input_app.input_train_neg_name]

  loss = trainer.build_train_graph(image_feature, text, neg_text)
  
  eval_names = ['loss']
  print('eval_names:', eval_names)
  
  test_flow(
    [loss], 
    names=eval_names,
    gen_feed_dict=input_app.gen_feed_dict,
    model_dir=FLAGS.model_dir, 
    num_interval_steps=FLAGS.num_interval_steps,
    num_epochs=FLAGS.num_epochs,
    eval_times=FLAGS.eval_times,
    sess=sess)

def main(_):
  logging.init(logtostderr=True, logtofile=False)
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    test()

if __name__ == '__main__':
  tf.app.run()

  
