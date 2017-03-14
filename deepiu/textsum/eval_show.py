#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval_show.py
#        \author   chenghuige  
#          \date   2016-08-27 17:05:57.683113
#   \Description  
# ==============================================================================

"""
 this is rigt now proper for bag of word model, show eval result
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_text_topn', 5, '')
flags.DEFINE_integer('num_word_topn', 50, '')

#---------for rnn decode
flags.DEFINE_integer('seq_decode_method', 0, """sequence decode method: 0 greedy, 1 sample, 2 full sample, 
                                                3 beam (beam search ingraph), 4 beam search (outgraph/interactive)
                                                Now only support greedy and beam search""")

import functools
import melt
from deepiu.image_caption import evaluator

from deepiu.util import text2ids
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

#-----------for show and tell textsum/seq2seq, all generated text method
def deal_eval_generated_texts_results(results):
  _, \
  evaluate_image_name, \
  evaluate_input_text_str, \
  evaluate_input_text, \
  evaluate_text_str, \
  evaluate_text, \
  generated_texts, \
  generated_texts_beam, \
  generated_texts_score, \
  generated_texts_score_beam, \
  pos_scores = results

  for i in xrange(len(evaluate_image_name)):
    #print(generated_texts_score_beam[i])
    #print(generated_texts_beam[i])
    evaluator.print_img_text_generatedtext_score(evaluate_image_name[i], i, 
                                      evaluate_input_text_str[i], 
                                      evaluate_input_text[i],
                                      evaluate_text_str[i], pos_scores[i],
                                      evaluate_text[i], 
                                      generated_texts[i], generated_texts_score[i],
                                      generated_texts_beam[i], generated_texts_score_beam[i])
  melt.print_results(results, ['eval_loss'])

def gen_eval_generated_texts_ops(input_app, input_results, predictor, eval_scores):
  #need distinct_texts.npy distinct_text_strs.npy
  evaluator.init()

  evaluate_image_name, evaluate_text, evaluate_text_str, \
  evaluate_input_text, evaluate_input_text_str = input_results[input_app.fixed_input_valid_name]
  num_evaluate_examples = input_app.num_evaluate_examples

  pos_scores = eval_scores[:num_evaluate_examples, 0]

  build_predict_text_graph = functools.partial(predictor.build_predict_text_graph,
                                               input_text=evaluate_input_text, 
                                               beam_size=FLAGS.beam_size, 
                                               convert_unk=False)

  generated_texts, generated_texts_score = build_predict_text_graph(
                      decode_method=FLAGS.seq_decode_method)

  #beam search(ingraph)
  generated_texts_beam, generated_texts_score_beam = build_predict_text_graph(
                      decode_method=SeqDecodeMethod.beam)

  eval_ops = [evaluate_image_name, evaluate_input_text_str, evaluate_input_text, \
              evaluate_text_str, evaluate_text, \
              generated_texts, generated_texts_beam, \
              generated_texts_score, generated_texts_score_beam, \
              pos_scores]

  print('eval_ops:')
  for eval_op in eval_ops:
    print(eval_op)

  return eval_ops
