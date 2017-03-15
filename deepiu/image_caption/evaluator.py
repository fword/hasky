#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluator.py
#        \author   chenghuige  
#          \date   2016-08-22 13:03:44.170552
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('valid_resource_dir', '/home/gezi/temp/image-caption/flickr/seq-with-unk/valid/', '')
flags.DEFINE_string('image_url_prefix', 'D:\\data\\\image-caption\\flickr\\imgs\\', 'http://b.hiphotos.baidu.com/tuba/pic/item/')
flags.DEFINE_string('label_file', '/home/gezi/data/image-caption/flickr/test/results_20130124.token', '')
flags.DEFINE_string('image_feature_file', '/home/gezi/data/image-caption/flickr/test/img2fea.txt', '')
flags.DEFINE_string('img2text', '', '')
flags.DEFINE_string('text2id', '', '')
flags.DEFINE_string('image_name_bin', '', '')
flags.DEFINE_string('image_feature_bin', '', '')

flags.DEFINE_integer('num_metric_eval_examples', 1000, '')
flags.DEFINE_integer('metric_eval_batch_size', 1000, '')
flags.DEFINE_integer('metric_topn', 100, 'only consider topn results when calcing metrics')

flags.DEFINE_integer('max_texts', 200000, '')

import sys, os
import gezi.nowarning

import gezi
import melt
logging = melt.logging

from deepiu.util import vocabulary
vocab = None
vocab_size = None
from deepiu.util.text2ids import ids2words, ids2text

from deepiu.image_caption.conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN
from deepiu.image_caption.input_app import texts2ids

import numpy as np
import math

all_distinct_texts = None
all_distinct_text_strs = None
image_labels = None

img2text = None
text2id = None

image_names = None
image_features = None

def init():
  #for evaluation without train will also use evaluator so only set log path in train.py
  #logging.set_logging_path(FLAGS.model_dir)
  test_dir = FLAGS.valid_resource_dir
  global all_distinct_texts, all_distinct_text_strs
  global vocab, vocab_size
  if all_distinct_texts is None:
    print('loading valid resorce from:', test_dir)
    vocabulary.init()
    vocab = vocabulary.vocab
    vocab_size = vocabulary.vocab_size
    
    if os.path.exists(test_dir + '/distinct_texts.npy'):
      all_distinct_texts = np.load(test_dir + '/distinct_texts.npy')
    else:
      all_distinct_texts = []
    
    #to avoid outof gpu mem
    all_distinct_texts = all_distinct_texts[:FLAGS.max_texts]
    print('all_distinct_texts len:', len(all_distinct_texts), file=sys.stderr)
    
    #--padd it as test data set might be smaller in shape[1]
    all_distinct_texts = np.array([gezi.nppad(text, TEXT_MAX_WORDS) for text in all_distinct_texts])
    if FLAGS.feed_dict:
      all_distinct_texts = texts2ids(evaluator.all_distinct_text_strs)
    if os.path.exists(test_dir + '/distinct_text_strs.npy'):
      all_distinct_text_strs = np.load(test_dir + '/distinct_text_strs.npy')
    else:
      all_distinct_text_strs = []


def init_labels():
  """
  assume to be in flickr format if is not npy binary file
  """
  global image_labels
  if image_labels is None:
    timer = gezi.Timer('init_labels from %s'%FLAGS.label_file)
    if FLAGS.label_file.endswith('.npy'):
      image_labels = np.load(FLAGS.label_file).item()
    else:
      image_labels = {}
      for line in open(FLAGS.label_file):
        l = line.rstrip().split('\t')
        img = l[0][:l[0].index('#')]
        text = l[-1]
        if img not in image_labels:
          image_labels[img] = set([text])
        else:
          image_labels[img].add(text)
    timer.print()

def get_image_labels():
  init_labels()
  return image_labels

def get_bidrectional_lable_map():
  global img2text, text2id
  if img2text is None:
    if FLAGS.img2text and FLAGS.text2id:
      img2text = np.load(FLAGS.img2text).item()
      text2id = np.load(FLAGS.text2id).item()
    else:
      img2text = text2id = {}
      for i, text in enumerate(all_distinct_text_strs):
        text2id[text] = i
      
      num_errors = 0
      print('label_file:', FLAGS.label_file)
      for line in open(FLAGS.label_file):
        l = line.rstrip().split('\t')
        img = l[0][:l[0].index('#')]
        text = l[-1].strip()
        if text not in text2id:
          #print(text)
          num_errors += 1
          continue
        id = text2id[text]

        if img not in img2text:
          img2text[img] = set([id])
        else:
          img2text[img].add(id)
      print('num_errors:', num_errors)
  return img2text, text2id

def get_label(img):
  image_labels = get_image_labels()
  return list(image_labels[img])[0]

def get_image_names_and_features():
  global image_names, image_features
  if image_names is None:
    timer = gezi.Timer('get_image_names_and_features')
    if FLAGS.image_name_bin and FLAGS.image_feature_bin:
      image_names = np.load(FLAGS.image_name_bin)
      image_features = np.load(FLAGS.image_feature_bin)
    else:
      lines = open(FLAGS.image_feature_file).readlines()
      image_names = np.array([line.split('\t')[0] for line in lines])
      image_features = np.array([[float(x) for x in line.split('\t')[1: 1 + IMAGE_FEATURE_LEN]] for line in lines])
    timer.print()
  return image_names, image_features

head_html = '<html><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><body>'
tail_html = '</body> </html>'

img_html = '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n {1} {2} epoch:{3}, step:{4}, train:{5}, eval:{6}, duration:{7}, {8}'
content_html = '<p> {} </p>'

import numpy as np

def print_neareast_texts(scores, num=20, img = None):
  indexes = (-scores).argsort()[:num]
  for i, index in enumerate(indexes):
    used_words = ids2words(all_distinct_texts[index])
    line = ' '.join([str(x) for x in ['%d:['%i,all_distinct_text_strs[index], ']', "%.6f"%scores[index], len(used_words), '/'.join(used_words)]])
    logging.info(content_html.format(line))

def print_neareast_words(scores, num=50):
  indexes = (-scores).argsort()[:num]
  line = ' '.join(['%s:%.6f'%(vocab.key(index), scores[index]) for index in indexes])
  logging.info(content_html.format(line))

def print_neareast_texts_from_sorted(scores, indexes, img = None):
  for i, index in enumerate(indexes):
    used_words = ids2words(all_distinct_texts[index])
    predict_result = ''
    if img:
      init_labels()
      if img in image_labels:
        predict_result = 'er@%d'%i if all_distinct_text_strs[index] not in image_labels[img] else 'ok@%d'%i
      else:
        predict_result = 'un@%d'%i 
    #notice may introduce error, offline scores is orinal scores! so need scores[index] but online input is max_scores will need scores[i]
    if len(scores) == len(indexes):
      line = ' '.join([str(x) for x in [predict_result, '[', all_distinct_text_strs[index], ']', "%.6f"%scores[i], len(used_words), '/'.join(used_words)]])
    else:
      line = ' '.join([str(x) for x in [predict_result, '[', all_distinct_text_strs[index], ']', "%.6f"%scores[index], len(used_words), '/'.join(used_words)]])
    logging.info(content_html.format(line))

def print_neareast_words_from_sorted(scores, indexes):
  if len(scores) == len(indexes):
    line = ' '.join(['%s:%.6f'%(vocab.key(int(index)), scores[i]) for i, index in enumerate(indexes)])
  else:
    line = ' '.join(['%s:%.6f'%(vocab.key(int(index)), scores[index]) for i, index in enumerate(indexes)])
  logging.info(content_html.format(line))

def print_img(img, i):
  img_url = FLAGS.image_url_prefix + img  
  logging.info(img_html.format(
    img_url, 
    i, 
    img, 
    melt.epoch(), 
    melt.step(), 
    melt.train_loss(), 
    melt.eval_loss(),
    melt.duration(),
    gezi.now_time()))

def print_img_text(img, i, text):
  print_img(img, i)
  logging.info(content_html.format(text))

def print_img_text_score(img, i, text, score):
  print_img(img, i)
  logging.info(content_html.format('{}:{}'.format(text, score)))

def print_img_text_negscore(img, i, text, score, text_ids, neg_text=None, neg_score=None, neg_text_ids=None):
  print_img(img, i)
  text_words = ids2text(text_ids)
  if neg_text is not None:
    neg_text_words = ids2text(neg_text_ids)
  logging.info(content_html.format('pos:[ {} ] {:.6f} {}'.format(text, score, text_words)))
  if neg_text is not None:
    logging.info(content_html.format('neg:[ {} ] {:.6f} {}'.format(neg_text, neg_score, neg_text_words)))  

#for show and tell 
def print_generated_text(generated_text, id=-1, name='gen'):
  if id >= 0:
    logging.info(content_html.format('{}_{}:[ {} ]'.format(name, id, ids2text(generated_text))))
  else:
    logging.info(content_html.format('{}:[ {} ]'.format(name, ids2text(generated_text))))

def print_generated_text_score(generated_text, score, id=-1, name='gen'):
  if id >= 0:
    logging.info(content_html.format('{}_{}:[ {} ] {:.6f}'.format(name, id, ids2text(generated_text), score)))
  else:
    logging.info(content_html.format('{}:[ {} ] {:.6f}'.format(name, ids2text(generated_text), score)))

def print_img_text_negscore_generatedtext(img, i, text, score,
                                          text_ids,  
                                          generated_text, generated_text_score,
                                          generated_text_beam=None, generated_text_score_beam=None,
                                          neg_text=None, neg_score=None, neg_text_ids=None):
  score = math.exp(-score)
  print_img_text_negscore(img, i, text, score, text_ids, neg_text, neg_score, neg_text_ids)
  try:
    print_generated_text_score(generated_text, generated_text_score)
  except Exception:
    for i, text in enumerate(generated_text):
      print_generated_text_score(text, generated_text_score[i], name='gen__max', id=i)   
  
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='gen_beam', id=i)


def print_img_text_generatedtext(img, i, input_text, input_text_ids, 
                                 text, score, text_ids,
                                 generated_text, generated_text_beam=None):
  print_img(img, i)
  score = math.exp(-score)
  input_text_words = ids2text(input_text_ids)
  text_words = ids2text(text_ids)
  logging.info(content_html.format('in_:[ {} ] {}'.format(input_text, input_text_words)))
  logging.info(content_html.format('pos:[ {} ] {:.6f} {}'.format(text, score, text_words)))
  print_generated_text(generated_text)
  if generated_text_beam is not None:
    print_generated_text(generated_text_beam)

def print_img_text_generatedtext_score(img, i, input_text, input_text_ids, 
                                 text, score, text_ids,
                                 generated_text, generated_text_score, 
                                 generated_text_beam=None, generated_text_score_beam=None):
  print_img(img, i)
  score = math.exp(-score)
  input_text_words = ids2text(input_text_ids)
  text_words = ids2text(text_ids)
  logging.info(content_html.format('in_:[ {} ] {}'.format(input_text, input_text_words)))
  logging.info(content_html.format('pos:[ {} ] {:.6f} {}'.format(text, score, text_words)))

  try:
    print_generated_text_score(generated_text, generated_text_score)
  except Exception:
    for i, text in enumerate(generated_text):
      print_generated_text_score(text, generated_text_score[i], name='gen__max', id=i)   
  
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='gen_beam', id=i)

score_op = None

def predicts(imgs, img_features, predictor, rank_metrics):
  timer = gezi.Timer('preidctor.bulk_predict')
  #score = predictor.bulk_predict(img_features, all_distinct_texts[:FLAGS.max_texts])
  # TODO gpu outofmem predict for showandtell
  score = predictor.bulk_predict(img_features, all_distinct_texts)
  timer.print()

  img2text, _ = get_bidrectional_lable_map()
  num_texts = all_distinct_texts.shape[0]

  for i, img in enumerate(imgs):
    indexes = (-score[i]).argsort()
    
    hits = img2text[img]

    num_positions = min(num_texts, FLAGS.metric_topn)
    labels = [indexes[j] in hits for j in xrange(num_positions)]

    rank_metrics.add(labels)

def evaluate_scores(predictor, random=False):
  timer = gezi.Timer('evaluate_scores')
  init()
  imgs, img_features = get_image_names_and_features()
  
  num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(imgs))
  step = FLAGS.metric_eval_batch_size

  if random:
    index = np.random.choice(len(imgs), num_metric_eval_examples, replace=False)
    imgs = imgs[index]
    img_features = img_features[index]

  text_max_words = all_distinct_texts.shape[1]
  rank_metrics = gezi.rank_metrics.RecallMetrics()

  print('text_max_words:', text_max_words)
  start = 0
  while start < num_metric_eval_examples:
    end = start + step
    if end > num_metric_eval_examples:
      end = num_metric_eval_examples
    print('predicts start:', start, 'end:', end, file=sys.stderr)
    predicts(imgs[start: end], img_features[start: end], predictor, rank_metrics)
    start = end
    
  melt.logging_results(
    rank_metrics.get_metrics(), 
    rank_metrics.get_names(), 
    tag='evaluate: epoch:{} step:{} loss:{} eval_loss:{}'.format(
      melt.epoch(), 
      melt.step(),
      melt.train_loss(),
      melt.eval_loss()))

  timer.print()

  return rank_metrics.get_metrics(), rank_metrics.get_names()
  
