#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-06 19:47:14.278205
#   \Description  
# ==============================================================================

"""
@FIXME will segmentaion fault at the end if using numpy
import numpy ...
so in virtual env, not use numpy if possible
or carefully test your import sequence

gezi must after libword_counter
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi.nowarning
import gezi  #since you introduce libgezi_util wihich use libstringutil or libgezi to be safe put at first
#from libword_counter import Vocabulary   #with this will core... with numpy...

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('vocab', './vocab.bin', 'vocabulary binary file')

#need resource of Segmentor or will check fail.. here not use segmentor for en
#need vocab file
#need model 

#input file of images
#input file of texts

#python predict.py text_file vocab model_dir 
#sys.stdin will be image feature file

import sys

import melt
logging = melt.logging

from deepiu.image_caption.algos import algos_factory
from deepiu.image_caption import vocabulary

import numpy as np


WORDS_SEP = ' '
TEXT_MAX_WORDS = 80
NUM_RESERVED_IDS = 1 
ENCODE_UNK = 0
IMAGE_FEATURE_LEN = 1000

algo = 'bow'

vocabulary.init()
vocab = vocabulary.vocab 

ids_list = []
text_list = []
for line in open('./test.txt'): 
  text = line.strip().split('\t')[-1]
  text_list.append(text)
  words = line.split()
  ids = [vocab.id(word) for word in text.split(WORDS_SEP) if vocab.has(word) or ENCODE_UNK]
  ids = gezi.pad(ids, TEXT_MAX_WORDS)
  ids_list.append(ids)
ids_list = np.array(ids_list)

def predict():
  predictor = algos_factory.gen_predictor(algo)
  predictor.init_predict(TEXT_MAX_WORDS)
  predictor.load('./model.ckpt-12000')

  for line in sys.stdin:
    l = line.strip().split('\t')
    image_name = l[0]
    image_feature = np.array([[float(x) for x in l[1:]]])
    #image_feature = [[float(x) for x in l[1:]]]

    scores = predictor.bulk_predict(image_feature, ids_list)[0]

    for i, score in enumerate(scores):
      print('{}\t{}\t{}'.format(image_name, score, text_list[i]))

  
def main(_):
  logging.init(logtostderr=True, logtofile=False)

  with tf.variable_scope(algo):
    predict()

if __name__ == '__main__':
  tf.app.run()
