#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-sim-score.py
#        \author   chenghuige  
#          \date   2016-09-25 00:46:53.890615
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'bow', 'default algo is cbow, @TODO lstm, cnn')
flags.DEFINE_string('model_dir', '/home/gezi/data/models/model.flickr.bow/', '')  

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('vocab', '/tmp/image-caption/flickr/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_integer('num_examples', 1000, '')
flags.DEFINE_integer('batch_size', 1000, '')

flags.DEFINE_string('image_feature_file', '/home/gezi/data/image-text-sim/flickr/test/img2fea.txt', '')

flags.DEFINE_integer('topn', 1, '')
flags.DEFINE_boolean('print_predict', False, '')

flags.DEFINE_string('out_file', 'sim-result.html', '')

import sys 
sys.path.append('../../')

import numpy as np
import melt
import gezi

import evaluator
evaluator.init()
melt.logging.init(file=FLAGS.out_file)

#algos
import algos.algos_factory
from algos.algos_factory import Algos

m = {}
for line in open(FLAGS.label_file):
  l = line.rstrip().split('\t')
  img = l[0][:l[0].index('#')]
  text = l[-1]

  if img not in m:
    m[img] = set([text])
  else:
    m[img].add(text)

f = open(FLAGS.image_feature_file).readlines()

print('total test images:', len(m))
print('total test texts:', evaluator.all_distinct_texts.shape[0])

metrics = ['mean_reciprocal_rank', \
           'r_precision', \
           'avg_precision', \
           'precision@1', \
           'precision@5', \
           'precision@10', \
           'precision@100', \
           'dcg@1', \
           'dcg@5', \
           'dcg@10', \
           'ndcg@1', \
           'ndcg@5', \
           'ndcg@10', \
           ]

hit = 0
total_hit = 0
def predicts(predictor, start, end):
  global hit, total_hit
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  img_features = np.array([[float(x) for x in line.split('\t')[1:]] for line in lines])
  score = predictor.bulk_predict(img_features, evaluator.all_distinct_texts)
  
  for i, img in enumerate(imgs):
    indexes = (-score[i]).argsort()
    
    if FLAGS.print_predict:
      evaluator.print_img(img, i)
      evaluator.print_neareast_texts_with_indexes(score[i], indexes, 20)

    has_hit = False
    for i in range(FLAGS.topn):
      if evaluator.all_distinct_text_strs[indexes[i]] in m[img]:
        total_hit += 1
        has_hit = True
    if has_hit:
      hit += 1
  print('hits:', hit)
  print('total_hits:', total_hit)

def evaluate_score():
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = algos.algos_factory.gen_predictor(FLAGS.algo)
  predictor.init_predict(text_max_words)
  predictor.load(FLAGS.model_dir)
  timer = gezi.Timer()
  start = 0
  while start < FLAGS.num_examples:
    end = start + FLAGS.batch_size
    if end > FLAGS.num_examples:
      end = FLAGS.num_examples
    print('predicts start:', start, 'end:', end, file=sys.stderr)
    predicts(predictor, start, end)
    start = end

  print('using time:', timer.elapsed())
  hit_ratio = hit / FLAGS.num_examples
  total_hit_ratio = total_hit / (FLAGS.num_examples * FLAGS.topn)
  print('num_hits:', hit)
  print('num_total_hits:', total_hit)
  print('hit_ratio:', hit_ratio)
  print('total_hit_ratio:', total_hit_ratio)

def main(_):
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    evaluate_score()
if __name__ == '__main__':
  tf.app.run()