#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   algos_factory.py
#        \author   chenghuige  
#          \date   2016-09-17 19:42:42.947589
#   \Description  
# ==============================================================================

"""
Should move to util
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import melt

from deepiu.image_caption.algos.bow import Bow, BowPredictor 
from deepiu.image_caption.algos.rnn import Rnn, RnnPredictor

from deepiu.image_caption.algos.show_and_tell import ShowAndTell 
from deepiu.image_caption.algos.show_and_tell_predictor import ShowAndTellPredictor

try:
  from deepiu.textsum.algos.seq2seq import Seq2seq, Seq2seqPredictor
  #from deepiu.textsum.alogs.seq2seq_attention import Seq2seqAttention
except Exception:
  pass

class Algos:
  bow = 'bow'    #bow encode for text
  rnn = 'rnn'    #rnn encode for text
  cnn = 'cnn'    #cnn encode for text
  show_and_tell = 'show_and_tell'   #lstm decode for text
  seq2seq = 'seq2seq'
  seq2seq_attention = 'seq2seq_attention'

class AlgosType:
   discriminant = 0
   generative = 1

AlgosTypeMap = {
  Algos.bow: AlgosType.discriminant,
  Algos.rnn: AlgosType.discriminant,
  Algos.show_and_tell: AlgosType.generative,
  Algos.seq2seq : AlgosType.generative,
  Algos.seq2seq_attention : AlgosType.generative
}

def is_discriminant(algo):
  return AlgosTypeMap[algo] == AlgosType.discriminant

def is_generative(algo):
  return AlgosTypeMap[algo] == AlgosType.generative

def _gen_builder(algo, is_predict=True):
  """
  Args:
  is_predict: set to False if only train, no need for predict/eval
  """
  if is_predict:
    if algo == Algos.bow:
      return BowPredictor()
    elif algo == Algos.show_and_tell:
      return ShowAndTellPredictor()
    elif algo == Algos.rnn:
      return RnnPredictor()
    elif algo == Algos.seq2seq:
      return Seq2seqPredictor()
    else:
      raise ValueError('Unsupported algo %s'%algo) 
  else:
    if algo == Algos.bow:
      return Bow()
    elif algo == Algos.show_and_tell:
      return ShowAndTell()
    elif algo == Algos.rnn:
      return Rnn()
    elif algo == Algos.seq2seq:
      return Seq2seq()
    else:
      raise ValueError('Unsupported algo %s'%algo) 

def gen_predictor(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    return _gen_builder(algo, is_predict=True)
  
def gen_tranier(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    return _gen_builder(algo, is_predict=False)

def gen_trainer_and_predictor(algo):
  trainer = gen_tranier(algo, reuse=None)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, predictor
