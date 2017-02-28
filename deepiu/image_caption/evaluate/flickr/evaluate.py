#!/usr/bin/env python
#coding=utf-8
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2016-07-25 16:38:18.123109
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('vocab', './data/train/vocab.bin', 'vocabulary binary file')
flags.DEFINE_string('model_dir', './model', '')

import sys
sys.path.append('../')
import numpy as np
import predictor
from predictor import Predictor
import melt
import gezi 
gezi.ENCODE = 'utf-8'

import evaluator

evaluator.init()
from evaluator import *

predictor = Predictor()
text_feature_final_npy = np.load('/tmp/train/text_feature.npy')
tf_score = predictor.build_fixed_text_graph(text_feature_final_npy)
words_feature_final_npy = np.load('/tmp/train/word_feature.npy')
melt.reuse_variables()
tf_word_score = predictor.build_fixed_text_graph(words_feature_final_npy)
sess = predictor.load(FLAGS.model_dir)
image_place = predictor.image_feature_place
model_path = predictor.model_path

num_examples = 1000
batch_size = 1000

f = open('/home/gezi/data/image-auto-comment/test/img2fea.txt').readlines()

def predicts(start, end):
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  features = np.array([[float(x) for x in line.split('\t')[1:]] for line in lines])
  score = sess.run(tf_score, {image_place: features})
  word_score = sess.run(tf_word_score, {image_place: features})
  for i, img in enumerate(imgs):
    print_img(img, i)
    print_neareast_texts(score[i])
    print_neareast_words(word_score[i])

start = 0
while start < num_examples:
  end = start + batch_size
  print('predicts start:', start, 'end:', end, file=sys.stderr)
  predicts(start, end)
  start = end