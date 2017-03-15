#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-19 06:54:26.594835
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

#FIXME: attention will hang..., no attention works fine
#flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model.seq2seq.attention/', '')
flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model.seq2seq/', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

flags.DEFINE_string('input_text_name', 'seq2seq/model_init_1/input_text:0', 'model_init_1 because predictor after trainer init')
flags.DEFINE_string('text_name', 'seq2seq/model_init_1/text:0', '')

import sys, os, math
import gezi, melt
import numpy as np

from deepiu.util import text2ids

import conf  
from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

#TODO: now copy from prpare/gen-records.py
def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method=FLAGS.seg_method, 
                               feed_single=FLAGS.feed_single, 
                               allow_all_zero=True, 
                               pad=False)

  word_ids = word_ids[:max_words]
  word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def predict(predictor, input_text, text):
  input_word_ids = _text2ids(input_text, INPUT_TEXT_MAX_WORDS)
  print('input_word_ids', input_word_ids, 'len:', len(input_word_ids))
  print(text2ids.ids2text(input_word_ids))
  word_ids = _text2ids(text, INPUT_TEXT_MAX_WORDS)
  print('word_ids', word_ids, 'len:', len(word_ids))
  print(text2ids.ids2text(word_ids))

  timer = gezi.Timer()
  score = predictor.inference(['score'], 
                              feed_dict= {
                                      FLAGS.input_text_name: [input_word_ids],
                                      FLAGS.text_name: [word_ids]
                                      })
  
  print('score:', score)
  print('calc score time(ms):', timer.elapsed_ms())

  timer = gezi.Timer()
  exact_score = predictor.inference(['exact_score'], 
                                    feed_dict= {
                                      FLAGS.input_text_name: [input_word_ids],
                                      FLAGS.text_name: [word_ids]
                                      })
  
  print('exact_score:', exact_score)
  print('calc score time(ms):', timer.elapsed_ms())

  timer = gezi.Timer()
  exact_prob, logprobs = predictor.inference(['exact_prob', 'seq2seq_logprobs'], 
                                    feed_dict= {
                                      FLAGS.input_text_name: [input_word_ids],
                                      FLAGS.text_name: [word_ids]
                                      })
  
  exact_prob = exact_prob[0]
  logprobs = logprobs[0]
  print('exact_prob:', exact_prob, 'ecact_logprob:', math.log(exact_prob))
  print('logprobs:', logprobs)
  print('sum_logprobs:', gezi.gen_sum_list(logprobs))
  print('calc prob time(ms):', timer.elapsed_ms())


def predicts(predictor, input_texts, texts):
  input_word_ids_list = [_text2ids(input_text, INPUT_TEXT_MAX_WORDS) for input_text in input_texts]
  word_ids_list = [_text2ids(text, INPUT_TEXT_MAX_WORDS) for text in texts]

  print(input_word_ids_list)
  print(word_ids_list)

  timer = gezi.Timer()
  score = predictor.inference(['score'], 
                              feed_dict= {
                                      FLAGS.input_text_name: input_word_ids_list,
                                      FLAGS.text_name: word_ids_list
                                      })
  
  print('score:', score)
  print('calc score time(ms):', timer.elapsed_ms())

  #TODO FIXME not work...  Incompatible shapes: [8] vs. [2,4]
  timer = gezi.Timer()
  exact_score = predictor.inference(['exact_score'], 
                                    feed_dict= {
                                      FLAGS.input_text_name: input_word_ids_list,
                                      FLAGS.text_name: word_ids_list
                                      })
  
  print('exact_score:', exact_score)
  print('calc score time(ms):', timer.elapsed_ms())

  timer = gezi.Timer()

  exact_prob, logprobs = predictor.inference(['exact_prob', 'seq2seq_logprobs'], 
                                    feed_dict= {
                                      FLAGS.input_text_name: input_word_ids_list,
                                      FLAGS.text_name: word_ids_list
                                      })
  
  print(exact_prob)
  print(logprobs)
  #print('exact_prob:', exact_prob, 'ecact_logprob:', math.log(exact_prob))
  #print('logprobs:', logprobs)
  #print('sum_logprobs:', gezi.gen_sum_list(logprobs))
  print('calc prob time(ms):', timer.elapsed_ms())



def main(_):
  text2ids.init()
  predictor = melt.Predictor(FLAGS.model_dir)
  #predict(predictor, '包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '蕾丝内裤女')
  #predict(predictor, '包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '性感内衣')
  #predict(predictor, '包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '性感女内裤')
  #predict(predictor, '包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '苹果电脑')
  #predict(predictor, '包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '性感透明内裤')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '蔬菜')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '橘子')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '辣椒种植')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '小辣椒图片')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '辣椒')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '小辣椒')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '辣椒辣椒')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '辣椒小辣椒')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '辣椒果实')
  predict(predictor, '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施', '小橘子')
  #predict(predictor, "学生迟到遭老师打 扇耳光揪头发把头往墙撞致3人住院", "女孩")
  #predict(predictor, "学生迟到遭老师打 扇耳光揪头发把头往墙撞致3人住院", "学生")
  #predict(predictor, "学生迟到遭老师打 扇耳光揪头发把头往墙撞致3人住院", "女生学生")
  #predict(predictor, "学生迟到遭老师打 扇耳光揪头发把头往墙撞致3人住院", "女生学术")

  predicts(predictor, 
           ['包邮买二送一性感女内裤低腰诱惑透视蕾丝露臀大蝴蝶三角内裤女夏-淘宝网', '大棚辣椒果实变小怎么办,大棚辣椒果实变小防治措施'],
           ['蕾丝内裤女', '辣椒种植'])
if __name__ == '__main__':
  tf.app.run()
