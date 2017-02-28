#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dump-text-feature.py
#        \author   chenghuige  
#          \date   2016-08-21 21:48:31.732052
#   \Description  
# ==============================================================================

"""
should be used after prepare/gen-distinct-texts.py
and train.py
with model ready
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'bow', 'default algo is cbow, @TODO lstm, cnn')
flags.DEFINE_string('model_dir', './model', '')  
flags.DEFINE_string('data_dir', '/tmp/train', '')  
flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')

flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('vocab', '/tmp/image-caption/flickr/train/vocab.bin', 'vocabulary binary file')

import sys 
sys.path.append('../../')

import numpy as np
import melt

#algos
import algos.algos_factory
from algos.algos_factory import Algos

def dump():
  text_npy = np.load(FLAGS.data_dir + '/distinct_texts.npy')
  predictor = algos.algos_factory.gen_predictor(FLAGS.algo)
  tf_text_feature = predictor.forward_fixed_text(text_npy)
  #sess = melt.load(FLAGS.model_dir) #if using this can not work with melt.constant @TODO
  sess = predictor.load(FLAGS.model_dir)
  text_feature = sess.run(tf_text_feature)
  np.save(FLAGS.data_dir + '/text_feature.npy', text_feature)
  #without below Exception AssertionError: AssertionError("Nesting violated for default stack of <type 'weakref'> objects",) in <bound method InteractiveSession.__del__ of <tensorflow.python.client.session.InteractiveSession object at 0x3ab0f10>> ignored
  sess.close()

def main(_):
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
    with tf.variable_scope(FLAGS.algo):
      dump()

if __name__ == '__main__':
  tf.app.run()
