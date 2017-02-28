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
flags.DEFINE_integer('seq_decode_method', 0, 'sequence decode method: 0 max prob, 1 sample, 2 full sample, 3 beam search')
flags.DEFINE_integer('beam_size', 5, 'for seq decode beam search size')

import melt
from deepiu.image_caption import evaluator

from deepiu.util import text2ids
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

def deal_eval_results(results):
  #eval_loss
  _,  \
  eval_max_score, \
  eval_max_index, \
  eval_word_max_score, \
  eval_word_max_index, \
  evaluate_image_name, \
  evaluate_text_str, \
  evaluate_text, \
  pos_scores, \
  neg_scores, \
  evaluate_neg_text_str, \
  evaluate_neg_text = results

  enumerate_list = enumerate(zip(
    evaluate_image_name, 
    evaluate_text_str, 
    pos_scores, 
    evaluate_text,
    evaluate_neg_text_str, 
    neg_scores, 
    evaluate_neg_text
    ))

  for i, (img, text, pos_score, text_ids, neg_text, neg_score, neg_text_ids) in enumerate_list:
    evaluator.print_img_text_negscore(img, i, text, pos_score, text_ids, neg_text, neg_score, neg_text_ids)
    evaluator.print_neareast_texts_from_sorted(eval_max_score[i], eval_max_index[i], img)
    evaluator.print_neareast_words_from_sorted(eval_word_max_score[i], eval_word_max_index[i])

  melt.print_results(results, ['loss'])


def gen_eval_show_ops(input_app, input_results, predictor, eval_scores, eval_neg_text, eval_neg_text_str):
  eval_ops = []
  #need distinct_texts.npy distinct_text_strs.npy
  evaluator.init()

  evaluate_image_name, evaluate_image_feature, evaluate_text, evaluate_text_str = input_results[input_app.fixed_input_valid_name]
  num_evaluate_examples = input_app.num_evaluate_examples

  all_distinct_texts = evaluator.all_distinct_texts 
  #print(all_distinct_texts[0], evaluator.all_distinct_text_strs[0], text2ids.ids2text(all_distinct_texts[0]))
  predictor.init_evaluate_constant_text(all_distinct_texts)
  pos_scores = eval_scores[:num_evaluate_examples, 0]
  neg_scores = eval_scores[:num_evaluate_examples, 1]
  #eval neg text strs
  evaluate_neg_text_str = eval_neg_text_str[:num_evaluate_examples, 0] 
  #eval neg text ids
  evaluate_neg_text = eval_neg_text[:num_evaluate_examples, 0, :]
  eval_score = predictor.build_evaluate_fixed_text_graph(evaluate_image_feature)
  eval_max_score, eval_max_index = tf.nn.top_k(eval_score, FLAGS.num_text_topn)
  eval_word_score = predictor.build_evaluate_image_word_graph(evaluate_image_feature)
  eval_word_max_score, eval_word_max_index = tf.nn.top_k(eval_word_score, FLAGS.num_word_topn)
  eval_ops += [eval_max_score, eval_max_index, eval_word_max_score, eval_word_max_index]
  eval_ops += [evaluate_image_name, evaluate_text_str, evaluate_text]
  #notice evaluate_neg_text_str will run here so every thing related must be placeholder to avoid running twice (if FLAGS.feed_dict=1 and FLAGS.show_eval=1)
  #@TODO at some steps we might want to use eval_score
  eval_ops += [pos_scores, neg_scores, evaluate_neg_text_str, evaluate_neg_text]

  print('eval_ops:')
  for eval_op in eval_ops:
    print(eval_op)

  return eval_ops

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
  melt.print_results(results, ['loss'])

def gen_eval_generated_texts_ops(input_app, input_results, predictor, eval_scores):
  #need distinct_texts.npy distinct_text_strs.npy
  evaluator.init()

  evaluate_image_name, evaluate_text, evaluate_text_str, \
  evaluate_input_text, evaluate_input_text_str = input_results[input_app.fixed_input_valid_name]
  num_evaluate_examples = input_app.num_evaluate_examples

  pos_scores = eval_scores[:num_evaluate_examples, 0]

  #pos_scores = tf.no_op()

  generated_texts, generated_texts_score = predictor.build_predict_text_graph(
                      evaluate_input_text, 
                      decode_method=FLAGS.seq_decode_method, 
                      beam_size=FLAGS.beam_size,
                      convert_unk=False)

  generated_texts_beam, generated_texts_score_beam = predictor.build_predict_text_graph(
                      evaluate_input_text, 
                      #decode_method=FLAGS.seq_decode_method, 
                      decode_method=SeqDecodeMethod.beam_search,  #beam search
                      beam_size=FLAGS.beam_size,
                      convert_unk=False)

  #generated_texts_beam = tf.no_op()

  eval_ops = [evaluate_image_name, evaluate_input_text_str, evaluate_input_text, \
              evaluate_text_str, evaluate_text, \
              generated_texts, generated_texts_beam, \
              generated_texts_score, generated_texts_score_beam, \
              pos_scores]

  print('eval_ops:')
  for eval_op in eval_ops:
    print(eval_op)

  return eval_ops


def deal_eval_generated_texts_results(results):
  ori_results = [x for x in results]

  #TODO better handle 9..
  if len(ori_results) == 9:
    ori_results += [None] * 3

  _, \
  evaluate_image_name, \
  evaluate_text_str, \
  evaluate_text, \
  generated_texts, \
  generated_texts_beam, \
  generated_texts_score, \
  generated_texts_score_beam, \
  pos_scores, \
  neg_scores, \
  evaluate_neg_text_str, \
  evaluate_neg_text = ori_results
 
  for i in xrange(len(evaluate_image_name)):
    if neg_scores is not None:
      evaluator.print_img_text_negscore_generatedtext(evaluate_image_name[i], i, 
                                      evaluate_text_str[i], pos_scores[i],
                                      evaluate_text[i], 
                                      generated_texts[i], generated_texts_score[i],
                                      generated_texts_beam[i], generated_texts_score_beam[i],
                                      evaluate_neg_text_str[i], 
                                      neg_scores[i],
                                      evaluate_neg_text[i])
    else:
      evaluator.print_img_text_negscore_generatedtext(evaluate_image_name[i], i, 
                                      evaluate_text_str[i], pos_scores[i],
                                      evaluate_text[i], 
                                      generated_texts[i], generated_texts_score[i],
                                      generated_texts_beam[i], generated_texts_score_beam[i])
  melt.print_results(results, ['loss'])

def gen_eval_generated_texts_ops(input_app, input_results, predictor, eval_scores, eval_neg_text=None, eval_neg_text_str=None):
  #need distinct_texts.npy distinct_text_strs.npy
  evaluator.init()

  evaluate_image_name, evaluate_image_feature, evaluate_text, evaluate_text_str = input_results[input_app.fixed_input_valid_name]
  num_evaluate_examples = input_app.num_evaluate_examples

  pos_scores = eval_scores[:num_evaluate_examples, 0]

  if eval_neg_text is not None:
    neg_scores = eval_scores[:num_evaluate_examples, 1]
    #eval neg text strs
    evaluate_neg_text_str = eval_neg_text_str[:num_evaluate_examples, 0] 
    #eval neg text ids
    evaluate_neg_text = eval_neg_text[:num_evaluate_examples, 0, :]

  generated_texts, generated_texts_score = predictor.build_predict_text_graph(
                      evaluate_image_feature, 
                      decode_method=FLAGS.seq_decode_method, 
                      beam_size=FLAGS.beam_size,
                      convert_unk=False)

  generated_texts_beam, generated_texts_score_beam = predictor.build_predict_text_graph(
                      evaluate_image_feature, 
                      #decode_method=FLAGS.seq_decode_method, 
                      decode_method=SeqDecodeMethod.beam_search,  #beam search
                      beam_size=FLAGS.beam_size,
                      convert_unk=False)

  eval_ops = [evaluate_image_name, evaluate_text_str, evaluate_text, \
              generated_texts, generated_texts_beam,
              generated_texts_score, generated_texts_score_beam, \
              pos_scores]
  
  if eval_neg_text is not None:
    eval_ops += [neg_scores, evaluate_neg_text_str, evaluate_neg_text]

  print('eval_ops:')
  for eval_op in eval_ops:
    print(eval_op)

  return eval_ops