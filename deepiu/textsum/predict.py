#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-09-02 10:52:36.566367
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model.seq2seq.attention/', '')

flags.DEFINE_string('algo', 'seq2seq', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')

flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

#----------strategy 
flags.DEFINE_boolean('pad', True, '')

flags.DEFINE_integer('beam_size', 10, 'for seq decode beam search size')


flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                      set to False incase you want to load some old model without algo scope''')

flags.DEFINE_string('global_scope', '', '')


import sys
import gezi
import melt
logging = melt.logging

from deepiu.image_caption.algos import algos_factory
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

#debug
from deepiu.util import vocabulary
from deepiu.util import text2ids

from deepiu.textsum.algos import seq2seq

import numpy as np


def main(_):
  text2ids.init()
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
 
  global sess
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)
  with tf.variable_scope(global_scope):
    predictor =  algos_factory.gen_predictor(FLAGS.algo)
    with tf.variable_scope('run'):
      text, text_score = predictor.init_predict_text(decode_method=SeqDecodeMethod.beam_search, 
                                                   beam_size=FLAGS.beam_size,
                                                   convert_unk=False)    
  predictor.load(FLAGS.model_dir) 
  predict(predictor, "王凯整容了吗_王凯整容前后对比照片")
  predict(predictor, "任达华传授刘德华女儿经 赞停工陪太太(图)")
  predict(predictor, "大小通吃汉白玉霸王貔貅摆件 正品开光镇宅招财")
  predict(predictor, "学生迟到遭老师打 扇耳光揪头发把头往墙撞致3人住院")

if __name__ == '__main__':
  tf.app.run()
