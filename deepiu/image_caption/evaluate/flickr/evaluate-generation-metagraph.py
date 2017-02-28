#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-generate.py
#        \author   chenghuige  
#          \date   2016-09-20 07:50:13.732992
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('model_dir', '../../model.flickr.show_and_tell2/', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/seq-with-unk/train/vocab.bin', 'vocabulary binary file')

#flags.DEFINE_string('image_feature_file', '/home/gezi/data/image-caption/flickr/test/img2fea.txt', '')

flags.DEFINE_string('algo', 'show_and_tell', '')

flags.DEFINE_integer('seq_decode_method', 0, 'sequence decode method: 0 max prob, 1 sample, 2 full sample, 3 beam search')
flags.DEFINE_integer('beam_size', 5, 'for seq decode beam search size')
#flags.DEFINE_boolean('ignore_unk', True, '')
flags.DEFINE_integer('num_images', 1000, '')
flags.DEFINE_integer('batch_size', 1000, '')

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('outfile', 'gen-result.html', '')

flags.DEFINE_string('image_feature_place', 'show_and_tell/model_init_1/image:0', '')

flags.DEFINE_boolean('use_label', True, '')

import sys 
sys.path.append('../../')

import numpy as np
import gezi
#@TODO must before text2ids otherwise raise AttributeError(name) AttributeError: num_samples
import algos.algos_factory
#import text2ids

import melt
logging = melt.logging

import evaluator

import text2ids 
from text2ids import idslist2texts

f = open(FLAGS.image_feature_file).readlines()


def predict(predictor, key, features):
  ids = predictor.inference(key, {FLAGS.image_feature_place: features})
  return idslist2texts(ids)

def predicts(predictor, start, end):
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  features = np.array([[float(x) for x in line.split('\t')[1:]] for line in lines])
  texts = predict(predictor, 'text', features)
  texts_without_unk = predict(predictor, 'text_nounk', features)

  for i, img in enumerate(imgs):
    if FLAGS.use_label:
      label = evaluator.get_label(img)
    logging.info('{} {}'.format(i, img))
    logging.info(gezi.img_html(FLAGS.image_url_prefix + imgs[i]))
    if FLAGS.use_label:
      logging.info(gezi.thtml('ori:%s'%label))
    logging.info(gezi.thtml(texts[i]))
    logging.info(gezi.thtml(texts_without_unk[i]))

def run():
  predictor = melt.Predictor(FLAGS.model_dir)
  
  logging.info('model:%s'%predictor.model_path)
  start = 0
  timer = gezi.Timer()
  while start < FLAGS.num_images:
    end = start + FLAGS.batch_size
    end = min(FLAGS.num_images, end)
    print('predicts start:', start, 'end:', end, file=sys.stderr)
    predicts(predictor, start, end)
    start = end
  print('time:', timer.elapsed())
  
def main(_):
  logging.init(FLAGS.outfile, mode='w')
  
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  else:
    global_scope = ''
  
  #with tf.variable_scope(FLAGS.algo):
  run()

if __name__ == '__main__':
  tf.app.run()
