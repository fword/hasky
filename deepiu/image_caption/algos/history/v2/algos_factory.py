#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   algos_factory.py
#        \author   chenghuige  
#          \date   2016-09-17 19:42:42.947589
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import algos.bow
import algos.bow_predictor  
import algos.show_and_tell
import algos.show_and_tell_predictor

import algos.rnn

import melt

class Algos:
  bow = 'bow'
  show_and_tell = 'show_and_tell'
  rnn = 'rnn'

class AlgosType:
   discriminant = 0
   generative = 1


AlgosTypeMap = {
  Algos.bow: AlgosType.discriminant,
  Algos.show_and_tell: AlgosType.generative,
  Algos.rnn: AlgosType.discriminant,
}

def is_discriminant_algo(algo):
  return AlgosTypeMap[algo] == AlgosType.discriminant

def is_generative(algo):
  return AlgosTypeMap[algo] == AlgosType.generative

def _gen_builder(algo, with_predict=True):
  """
  Args:
  with_predict: set to False if only train, no need for predict/eval
  """
  if with_predict:
    if algo == Algos.bow:
      return algos.bow_predictor.BowPredictor()
    elif algo == Algos.show_and_tell:
      return algos.show_and_tell_predictor.ShowAndTellPredictor()
    elif algo == Algos.rnn:
      print('Warning: only trainer rnn trainer is ok right now')
      return None
    else:
      raise ValueError('Unsupported algo %s'%algo) 
  else:
    if algo == Algos.bow:
      return algos.bow.Bow()
    elif algo == Algos.show_and_tell:
      return algos.show_and_tell.ShowAndTell()
    elif algo == Algos.rnn:
      return algos.rnn.Rnn()
    else:
      raise ValueError('Unsupported algo %s'%algo) 

def gen_predictor(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    return _gen_builder(algo, with_predict=True)
  
def gen_tranier(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    return _gen_builder(algo, with_predict=False)

def gen_trainer_and_predictor(algo):
  trainer = gen_tranier(algo, reuse=None)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, predictor