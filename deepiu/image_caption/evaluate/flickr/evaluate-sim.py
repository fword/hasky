#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-sim-score.py
#        \author   chenghuige  
#          \date   2016-09-25 00:46:53.890615
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'bow', 'bow, rnn, show_and_tell')
flags.DEFINE_string('model_dir', '../../model.flickr.bow/', '')  

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/bow/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_boolean('print_predict', False, '')
#flags.DEFINE_string('out_file', 'sim-result.html', '')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi

from deepiu.image_caption import evaluator
from deepiu.image_caption.algos import algos_factory


def evaluate_score():
  evaluator.init()
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = algos_factory.gen_predictor(FLAGS.algo)
  predictor.init_predict(text_max_words)
  predictor.load(FLAGS.model_dir)

  evaluator.evaluate_scores(predictor)


def main(_):
  logging.init(logtostderr=True, logtofile=False)
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    evaluate_score()

if __name__ == '__main__':
  tf.app.run()
