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


import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('vocab', './vocab.bin', 'vocabulary binary file')
flags.DEFINE_string('image_feature_place', 'image_feature:0', '')
flags.DEFINE_string('text_place', 'text:0', '')
flags.DEFINE_string('algo', 'bow', '')

import sys

import numpy as np

#import gezi.nowarning
import gezi  #since you introduce libgezi_util wihich use libstringutil or libgezi to be safe put at first

import melt
logging = melt.logging

#from deepiu.image_caption import vocabulary
from libword_counter import Vocabulary   #with this will core... with numpy...

WORDS_SEP = ' '
TEXT_MAX_WORDS = 80
NUM_RESERVED_IDS = 1 
ENCODE_UNK = 0
IMAGE_FEATURE_LEN = 1000

predictor = melt.Predictor('./model.ckpt-12000')

#vocabulary.init()
#vocab = vocabulary.vocab 
vocab = Vocabulary(FLAGS.vocab, NUM_RESERVED_IDS)

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


def bulk_predict(predictor, images, texts):
  scores = predictor.inference('score', 
                               { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images,
                                 '%s/%s'%(FLAGS.algo, FLAGS.text_place): texts })
  return scores

def predict():
  for line in sys.stdin:
    l = line.strip().split('\t')
    image_name = l[0]
    image_feature = np.array([[float(x) for x in l[1:]]])
    #image_feature = [[float(x) for x in l[1:]]]

    scores = bulk_predict(predictor, image_feature, ids_list)[0]

    for i, score in enumerate(scores):
      print('{}\t{}\t{}'.format(image_name, score, text_list[i]))

  
def main(_):
  logging.init(logtostderr=True, logtofile=False)
  predict()

if __name__ == '__main__':
  tf.app.run()
