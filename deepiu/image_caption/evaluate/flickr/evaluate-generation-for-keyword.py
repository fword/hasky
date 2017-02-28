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
  
flags.DEFINE_string('model_dir', '/home/gezi/temp/image-caption/model.flickr.showandtell.for_keyword/', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/bow.for_keyword/train/vocab.bin', 'vocabulary binary file')

#flags.DEFINE_string('image_feature_file', '/home/gezi/data/image-caption/keyword/manyidu-2016q2/img2fea.txt', '')

flags.DEFINE_integer('seq_decode_method', 0, 'sequence decode method: 0 max prob, 1 sample, 2 full sample, 3 beam search')

flags.DEFINE_integer('beam_size', 5, 'for seq decode beam search size')
#flags.DEFINE_boolean('ignore_unk', True, '')
flags.DEFINE_integer('num_images', 1000, '')
#flags.DEFINE_integer('batch_size', 1000, '')

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('outfile', 'gen-result.html', '')

flags.DEFINE_string('image_feature_place', 'model_init_1/image_feature:0', '')
flags.DEFINE_string('text_place', 'text:0', '')
flags.DEFINE_string('algo', 'show_and_tell', '')

flags.DEFINE_boolean('use_label', False, '')

import sys 
#sys.path.append('../../')

import numpy as np
import gezi
#@TODO must before text2ids otherwise raise AttributeError(name) AttributeError: num_samples
from deepiu.image_caption.algos import algos_factory
#import text2ids

import melt
logging = melt.logging

from deepiu.image_caption import evaluator
from deepiu.image_caption import text2ids
from deepiu.image_caption.text2ids import idslist2texts

import conf  
from conf import IMAGE_FEATURE_LEN

f = open(FLAGS.image_feature_file).readlines()

def bulk_predict(predictor, images, texts):
  feed_dict = { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images,
                '%s/%s'%(FLAGS.algo, FLAGS.text_place): texts }
  #---setting to None you can get from erorr log what need to feed
  #feed_dict = None
  #feed_dict = { '%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): images }
  scores = predictor.inference('score', feed_dict)
  scores = scores.reshape((len(texts),))
  return scores

def sort_print(keywords, scores):
  indexes = (-np.array(scores)).argsort()
  results = '\t'.join(['{}:{}'.format(keywords[index], scores[index]) for index in indexes])
  logging.info(gezi.thtml(results))

def _predict(predictor, feature, keywords):
  ids = text2ids.texts2ids(keywords, 'en', False)
  feature = np.array([feature] * len(ids))
  scores = bulk_predict(predictor, feature, ids)
  sort_print(keywords, scores)


def predict(predictor, key, features):
  ids = predictor.inference(key, {'%s/%s'%(FLAGS.algo, FLAGS.image_feature_place): features})
  #ids = predictor.inference(key, {})
  return idslist2texts(ids)

def predicts(predictor, start, end):
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  features = np.array([[float(x) for x in line.split('\t')[1: 1 + IMAGE_FEATURE_LEN]] for line in lines])
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

    keywords = ['A', 'Two', 'Three', 'A group']
    _predict(predictor, features[i], keywords)


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

  text2ids.init()
  
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  else:
    global_scope = ''
  
  #with tf.variable_scope(FLAGS.algo):
  run()

if __name__ == '__main__':
  tf.app.run()
