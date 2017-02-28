#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-sim-metagraph.py
#        \author   chenghuige  
#          \date   2016-10-03 07:53:50.487381
#   \Description  
# ==============================================================================
"""
can run but result wrong, @FIXME well ok le.. , why fixed..
now only evaluate-genearte-metagrahph.py ok
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('model_dir', '', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/seq-with-unk/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_string('algo', 'show_and_tell', '')

flags.DEFINE_string('outfile', 'sim-result.html', '')

flags.DEFINE_string('image_feature_place', 'image_feature:0', '')
flags.DEFINE_string('text_place', 'text:0', '')

flags.DEFINE_boolean('use_label', True, '')

flags.DEFINE_boolean('print_predict', False, '')
flags.DEFINE_integer('batch_size', 1000, '')

import sys 
sys.path.append('../../')

import numpy as np
import gezi

import sys 
sys.path.append('../../')

import numpy as np
import melt
logging = melt.utils.logging
import gezi

import conf 
from conf import TEXT_MAX_WORDS

import evaluator
evaluator.init()

img2text, _ = evaluator.get_bidrectional_lable_map()

f = open(FLAGS.image_feature_file).readlines()

def bulk_predict(predictor, images, texts):
  scores = predictor.inference('score', 
                               { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images,
                                 '%s/%s'%(FLAGS.algo, FLAGS.text_place): texts })
  #scores = predictor.inference('score')
  return scores

print('total test images:', len(img2text))
print('total test texts:', evaluator.all_distinct_texts.shape[0])

rank_metrics = gezi.rank_metrics.RankMetrics()
num_texts = evaluator.all_distinct_texts.shape[0]

#do this already in evaluator.py
#all_distinct_texts = np.array([gezi.nppad(text, TEXT_MAX_WORDS) for text in evaluator.all_distinct_texts])
all_distinct_texts = evaluator.all_distinct_texts
print(all_distinct_texts.shape)
#print(all_distinct_texts)

def predicts(predictor, start, end):
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  img_features = np.array([[float(x) for x in line.split('\t')[1:]] for line in lines])
  score = bulk_predict(predictor, img_features, all_distinct_texts)
  #print(score)
  for i, img in enumerate(imgs):
    indexes = (-score[i]).argsort()
    
    if FLAGS.print_predict:
      #print(i, img)
      evaluator.print_img(img, i)
      evaluator.print_neareast_texts_from_sorted(score[i], indexes[:10], img)
      #evaluator.print_neareast_texts(score[i], num=20, img = None)

    hits = img2text[img]
    labels = [indexes[j] in hits for j in xrange(num_texts)]

    rank_metrics.add(labels)

def evaluate_score():
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = melt.Predictor(FLAGS.model_dir)
  timer = gezi.Timer()
  start = 0
  while start < FLAGS.num_examples:
    end = start + FLAGS.batch_size
    if end > FLAGS.num_examples:
      end = FLAGS.num_examples
    print('predicts start:', start, 'end:', end, file=sys.stderr)
    predicts(predictor, start, end)
    start = end
    
  melt.print_results(rank_metrics.get_metrics(), rank_metrics.get_names())
  print('predict using time:', timer.elapsed())
  
def main(_):
  logging.init(FLAGS.outfile, mode='w')

  assert FLAGS.model_dir
  
  evaluate_score()

if __name__ == '__main__':
  tf.app.run()
