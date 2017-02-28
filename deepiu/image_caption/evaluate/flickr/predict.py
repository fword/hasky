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
  
flags.DEFINE_string('model_dir', '../model.lstm/', '')
flags.DEFINE_string('img_feature_file', '/home/gezi/data/image-auto-comment/test/img2fea.txt', '')
flags.DEFINE_integer('num_images', 10, '')
flags.DEFINE_string('image_url_prefix', 'http://b.hiphotos.baidu.com/tuba/pic/item/', '')

#----------strategy 
flags.DEFINE_boolean('pad', True, '')
flags.DEFINE_string('seg_method', 'phrase', '')
flags.DEFINE_boolean('feed_single', True, '')


import sys 
sys.path.append('../')

import numpy as np

import gezi

import algos.show_and_tell
import algos.show_and_tell_predictor
from algos.show_and_tell_predictor import ShowAndTellPredictor as Predictor

import text2ids

predictor = Predictor()
predictor.init_predict()
predictor.load(FLAGS.model_dir)

texts = ['好吃', '帅哥', '美女', '胸大', '漂亮', '风景好', '小孩', '可爱' , \
         '画得好', '666', '面条', '花', '丑', '不丑', '好看', '不好看', \
         '不漂亮', '超漂亮', '性感', '超性感', '不性感', '美腿']


img_names = []
features = []

for line in open(FLAGS.img_feature_file):
  l = line.split('\t')
  img_name = l[0]
  feature = l[1:]
  img_names.append(img_name)
  features.append(feature)

  if len(features) == FLAGS.num_images:
    break

losses = predictor.bulk_predict(features,
                                text2ids.texts2ids(texts, 
                                                   pad=FLAGS.pad, 
                                                   seg_method=FLAGS.seg_method, 
                                                   feed_single=FLAGS.feed_single))
#print(losses)
seg_texts = text2ids.texts2segtexts(texts, 
                               pad=FLAGS.pad, 
                               seg_method=FLAGS.seg_method, 
                               feed_single=FLAGS.feed_single)

for img_name, loss in zip(img_names, losses):
  gezi.imgprint(FLAGS.image_url_prefix + img_name)
  indexes = loss.argsort()
  #print('indexes:', indexes)
  for index in indexes:
    print('[%s]'%texts[index], loss[index], seg_texts[index])
