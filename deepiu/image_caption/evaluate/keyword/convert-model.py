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

flags.DEFINE_string('algo', 'show_and_tell', 'bow, rnn, show_and_tell')
flags.DEFINE_string('model_dir', '/home/gezi/work/keywords/train/v2/zhongce/models/showandtell', '')  

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('vocab', '/home/gezi/work/keywords/train/v2/zhongce/models/showandtell/vocab.bin', 'vocabulary binary file')

flags.DEFINE_boolean('print_predict', False, '')
#flags.DEFINE_string('out_file', 'sim-result.html', '')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi

from deepiu.image_caption import evaluator
from deepiu.image_caption.algos import algos_factory

import conf 
from conf import TEXT_MAX_WORDS

#NOTICE converted by tf 11, then can not used in tf 10 ..,using meta data load
def evaluate_score():
  predictor = algos_factory.gen_predictor(FLAGS.algo)
  score = predictor.init_predict(TEXT_MAX_WORDS)
  tf.add_to_collection('score', score)
  predictor.load(FLAGS.model_dir)
  step = melt.get_model_step_from_dir(FLAGS.model_dir) 
  model_dir, _ = melt.get_model_dir_and_path(FLAGS.model_dir)
  print('step', step, file=sys.stderr)
  print('model_dir', model_dir)
  #melt.save_model(melt.get_session(), FLAGS.model_dir, step + 1)
  melt.save_model(melt.get_session(), model_dir, step + 1)
  #melt.get_session().close()

def main(_):
  logging.init(logtostderr=True, logtofile=False)
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    evaluate_score()

if __name__ == '__main__':
  tf.app.run()
