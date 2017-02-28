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
  
flags.DEFINE_string('m1', '', '')
flags.DEFINE_string('m2', '', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/image-caption/flickr/seq-with-unk/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_string('img_feature_file', '/home/gezi/data/image-text-sim/flickr/test/img2fea.txt', '')

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

tf.app.flags.DEFINE_integer('index', 0,
                            """out index of the second checkpoints""")

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

f = open(FLAGS.img_feature_file).readlines()


def predicts(predictor1, predictor2, start, end):
  lines = f[start:end]
  imgs = [line.split('\t')[0] for line in lines]
  features = np.array([[float(x) for x in line.split('\t')[1:]] for line in lines])
  texts = predictor1.predict_text(features, 0)
  texts_without_unk = predictor1.predict_text(features, 1)
  
  texts2 = predictor2.predict_text(features, 0)
  texts2_without_unk = predictor2.predict_text(features, 1)

  for i, img in enumerate(imgs):
    label = evaluator.get_label(img)
    logging.info('{} {}'.format(i, img))
    logging.info(gezi.img_html(FLAGS.image_url_prefix + imgs[i]))
    logging.info(gezi.thtml('ori:%s'%label))
    
    logging.info(gezi.thtml(texts2[i]))
    logging.info(gezi.thtml(texts2_without_unk[i]))
    
    logging.info(gezi.thtml(texts[i]))
    logging.info(gezi.thtml(texts_without_unk[i]))


global_scope = None
global_scope2 = None

def create_predictor(model_path, scope):
  print('scope:', scope)
  print('now scope:', tf.get_variable_scope().name)
  with tf.variable_scope(scope):
    predictor = algos.algos_factory.gen_predictor(FLAGS.algo)
    predictor.init_predict_text(FLAGS.seq_decode_method, FLAGS.beam_size, ignore_unk=False)
    melt.reuse_variables()
    predictor.init_predict_text(FLAGS.seq_decode_method, FLAGS.beam_size, ignore_unk=True)
    predictor.load(model_path, var_list=melt.variables_with_scope(scope))
    return predictor

def run():
  predictor1 = create_predictor(FLAGS.m1, global_scope)
  predictor2 = create_predictor(FLAGS.m2, global_scope2)

  logging.info('model1:%s'%predictor1.model_path)
  logging.info('model2:%s'%predictor2.model_path)
  start = 0
  timer = gezi.Timer()
  while start < FLAGS.num_images:
    end = start + FLAGS.batch_size
    end = min(FLAGS.num_images, end)
    print('predicts start:', start, 'end:', end, file=sys.stderr)
    predicts(predictor1, predictor2, start, end)
    start = end
  print('time:', timer.elapsed())
  
def main(_):
  logging.init(FLAGS.outfile, mode='w')
  global global_scope, global_scope2
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  else:
    global_scope = ''
  
  global_scope2 = '%s_%d'%(global_scope, FLAGS.index)

  run()

if __name__ == '__main__':
  tf.app.run()
