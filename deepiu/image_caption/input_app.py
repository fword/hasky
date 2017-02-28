#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   input_app.py
#        \author   chenghuige  
#          \date   2016-08-27 14:58:25.032042
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

import melt
logging = melt.logging
from deepiu.image_caption import input
import gezi
list_files = gezi.bigdata_util.list_files

#from deepiu.image_caption.conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN
import conf 
from conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN
from deepiu.util import text2ids 

texts2ids = None
def init():
  global texts2ids
  if texts2ids is None:
    texts2ids = lambda x: text2ids.texts2ids(x, 
                                             seg_method=FLAGS.seg_method, 
                                             feed_single=FLAGS.feed_single)

def text_placeholder(name):
  #@TODO make [None, None], but not ergent, also write records can change to sparse without padding, so more flexible
  #'last dimension shape must be known but is None'
  #so if use feed mode must set fixed length, may be use buket like tf seq2seq example @TODO
  return tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name=name) 

def image_feature_placeholder(name):
  return tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name=name) 

def image_name_placeholder(name):
  return tf.placeholder(tf.string, None, name=name)

def text_str_placeholder(name):
  return tf.placeholder(tf.string, None, name=name)

def print_input_results(input_results):
  print('input_results:')
  for name, tensors in input_results.items():
    print(name)
    if tensors:
      for tensor in tensors:
        print(tensor)

class InputApp(object):
  def __init__(self):
    self.input_train_name = 'input_train'
    self.input_train_neg_name = 'input_train_neg'
    self.input_valid_name = 'input_valid'
    self.fixed_input_valid_name = 'fixed_input_valid'
    self.input_valid_neg_name = 'input_valid_neg'
    #-----------common for all app inputs may be 
    #for evaluate show small results, not for evaluate show cost
    self.num_records = None
    self.num_evaluate_examples = None
    self.num_steps_per_epoch = None

    self.sess = melt.get_session()

    self.train_with_validation = None
    self.eval_fixed_images = None

    # self.step = 0

  def gen_feed_dict(self):
    feed_dict = {}
    if FLAGS.feed_dict:
      image_feature, text_str, neg_text_str = \
        self.sess.run([self.image_feature, self.text_str, self.neg_text_str])
      feed_dict = {
        self.image_feature_place: image_feature, 
        self.text_place: texts2ids(text_str),
        self.neg_text_place: texts2ids(neg_text_str)
        }
      
      #print(texts2ids(text_str)[0], len(text_str), text_str[0], text2ids.ids2text(texts2ids(text_str)[0]))
      #print(texts2ids(text_str).shape)
      #print(texts2ids(neg_text_str)[0], len(neg_text_str), neg_text_str[0], text2ids.ids2text(texts2ids(neg_text_str)[0]))
      #print(texts2ids(neg_text_str).shape)
  
      # if self.step == 42:
      #   for text, text_str, seg_text in zip(texts2ids(neg_text_str), neg_text_str, text2ids.idslist2texts(texts2ids(neg_text_str))):
      #     print(text, text_str, seg_text)
      # self.step += 1

    return feed_dict

  #but when eval will also need to feed those in gen_feed_dict..
  def gen_eval_feed_dict(self):
    eval_feed_dict = {}
    if FLAGS.feed_dict:
      if self.train_with_validation:
        if not self.eval_fixed_images:
          eval_image_feature, eval_text_str, eval_neg_text_str = \
            self.sess.run([self.eval_image_feature, self.eval_text_str, self.eval_neg_text_str])
          eval_feed_dict = {
            self.image_feature_place: eval_image_feature,
            self.text_place: texts2ids(eval_text_str),
            self.neg_text_place: texts2ids(eval_neg_text_str)
            }
        else:
          eval_image_feature, eval_text_str, eval_neg_text_str, \
          fixed_image_feature, fixed_text_str,  \
          eval_image_name, eval_text_str,  \
          fixed_image_name, fixed_text_str, \
          eval_neg_text_str \
            = self.sess.run([self.eval_image_feature, self.eval_text_str, self.eval_neg_text_str, \
                             self.fixed_image_feature, self.fixed_text_str, \
                             self.eval_image_name, self.eval_text_str, \
                             self.fixed_image_name, self.fixed_text_str, \
                             self.eval_neg_text_str])

          eval_feed_dict = {
            self.image_feature_place: eval_image_feature,
            self.text_place: texts2ids(eval_text_str),
            self.neg_text_place: texts2ids(eval_neg_text_str),
            self.fixed_image_feature_place: fixed_image_feature,
            self.fixed_text_place: texts2ids(fixed_text_str),
            self.eval_image_name_place: eval_image_name,
            self.eval_text_str_place: eval_text_str,
            self.fixed_image_name_place: fixed_image_name,
            self.fixed_text_str_place: fixed_text_str,
            self.eval_neg_text_str_place: eval_neg_text_str,
            }
         
    return eval_feed_dict

  def gen_train_input(self, inputs, decode):
     #--------------------- train
    logging.info('train_input: %s'%FLAGS.train_input)
    trainset = list_files(FLAGS.train_input)
    logging.info('trainset:{} {}'.format(len(trainset), trainset[:2]))
    
    assert len(trainset) >= FLAGS.min_records, len(trainset)
    if FLAGS.num_records > 0:
      assert len(trainset) == FLAGS.num_records, len(trainset)

    num_records = gezi.read_int_from(FLAGS.num_records_file)
    logging.info('num_records:{}'.format(num_records))
    logging.info('batch_size:{}'.format(FLAGS.batch_size))
    logging.info('FLAGS.num_gpus:{}'.format(FLAGS.num_gpus))
    num_gpus = max(FLAGS.num_gpus, 1)
    num_steps_per_epoch = num_records // (FLAGS.batch_size * num_gpus)
    logging.info('num_steps_per_epoch:{}'.format(num_steps_per_epoch))
    self.num_records = num_records
    self.num_steps_per_epoch = num_steps_per_epoch
    
    image_name, image_feature, text, text_str = inputs(
      trainset, 
      decode=decode,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      #seed=seed,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle,
      fix_sequence=FLAGS.fix_sequence,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      name=self.input_train_name)

    if FLAGS.feed_dict:
      self.text_place =  text_placeholder('text_place')
      self.text_str = text_str
      text = self.text_place

      self.image_feature_place = image_feature_placeholder('image_feature_place') 
      self.image_feature = image_feature
      image_feature = self.image_feature_place
    
    if FLAGS.monitor_level > 1:
      lengths = melt.length(text)
      melt.scalar_summary("text/batch_min", tf.reduce_min(lengths))
      melt.scalar_summary("text/batch_max", tf.reduce_max(lengths))
      melt.scalar_summary("text/batch_mean", tf.reduce_mean(lengths))
    return (image_name, image_feature, text, text_str), trainset

  def gen_train_neg_input(self, inputs, decode_neg, trainset):
    neg_text, neg_text_str = inputs(
      trainset, 
      decode=decode_neg,
      batch_size=FLAGS.batch_size * FLAGS.num_negs,
      num_epochs=0, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      #fix_sequence=FLAGS.fix_sequence,
      name=self.input_train_neg_name)

    if FLAGS.feed_dict:
      self.neg_text_place = text_placeholder('neg_text_place')
      self.neg_text_str = neg_text_str
      neg_text = self.neg_text_place

    neg_text, neg_text_str = input.reshape_neg_tensors(
      [neg_text, neg_text_str], FLAGS.batch_size, FLAGS.num_negs)

    return neg_text, neg_text_str

  def gen_valid_input(self, inputs, decode):
    #---------------------- valid  
    validset = list_files(FLAGS.valid_input)
    logging.info('validset:{} {}'.format(len(validset), validset[:2]))
    eval_image_name, eval_image_feature, eval_text, eval_text_str = inputs(
      validset, 
      decode=decode,
      batch_size=FLAGS.eval_batch_size,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.eval_shuffle,
      seed=FLAGS.eval_seed,
      fix_random=FLAGS.eval_fix_random,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      fix_sequence=FLAGS.fix_sequence,
      name=self.input_valid_name)
    if FLAGS.feed_dict:
      self.eval_text_str = eval_text_str
      eval_text = self.text_place

      self.eval_image_feature = eval_image_feature
      eval_image_feature = self.image_feature_place

      if FLAGS.show_eval:
        self.eval_image_name_place = image_name_placeholder('eval_image_name_place')
        self.eval_image_name = eval_image_name
        eval_image_name = self.eval_image_name_place

        self.eval_text_str_place = text_str_placeholder('eval_text_str_place')
        self.eval_text_str = eval_text_str
        eval_text_str = self.eval_text_str_place

    eval_batch_size = FLAGS.eval_batch_size
   
    eval_result = eval_image_name, eval_image_feature, eval_text, eval_text_str
    eval_show_result = None
    
    if FLAGS.show_eval:
      eval_fixed_images = bool(FLAGS.fixed_valid_input)
      self.eval_fixed_images = eval_fixed_images
      if eval_fixed_images:
        assert FLAGS.fixed_eval_batch_size >= FLAGS.num_fixed_evaluate_examples, '%d %d'%(FLAGS.fixed_eval_batch_size, FLAGS.num_fixed_evaluate_examples)
        logging.info('fixed_eval_batch_size:{}'.format(FLAGS.fixed_eval_batch_size))
        logging.info('num_fixed_evaluate_examples:{}'.format(FLAGS.num_fixed_evaluate_examples))
        logging.info('num_evaluate_examples:{}'.format(FLAGS.num_evaluate_examples))
        #------------------- fixed valid
        fixed_validset = list_files(FLAGS.fixed_valid_input)
        logging.info('fixed_validset:{} {}'.format(len(fixed_validset), fixed_validset[:2]))
        fixed_image_name, fixed_image_feature, fixed_text, fixed_text_str = inputs(
          fixed_validset, 
          decode=decode,
          batch_size=FLAGS.fixed_eval_batch_size,
          fix_sequence=True,
          num_prefetch_batches=FLAGS.num_prefetch_batches,
          min_after_dequeue=FLAGS.min_after_dequeue,
          name=self.fixed_input_valid_name)
        
        if FLAGS.feed_dict:
          self.fixed_text_place = text_placeholder('fixed_text_place')
          self.fixed_text_str = fixed_text_str
          fixed_text = self.fixed_text_place
          
          self.fixed_image_feature_place = image_feature_placeholder('fixed_image_feature_place')
          self.fixed_image_feature = fixed_image_feature
          fixed_image_feature = self.fixed_image_feature_place
          
          self.fixed_image_name_place = image_name_placeholder('fixed_image_name_place')
          self.fixed_image_name = fixed_image_name
          fixed_image_name = self.fixed_image_name_place

          self.fixed_text_str_place = text_str_placeholder('fixed_text_str_place')
          self.fixed_text_str = fixed_text_str
          fixed_text_str = self.fixed_text_str_place

        #-------------shrink fixed image input as input batch might large then what we want to show, only choose top num_fixed_evaluate_examples
        fixed_image_name = melt.first_nrows(fixed_image_name, FLAGS.num_fixed_evaluate_examples)
        fixed_image_feature = melt.first_nrows(fixed_image_feature, FLAGS.num_fixed_evaluate_examples)
        fixed_text = melt.first_nrows(fixed_text, FLAGS.num_fixed_evaluate_examples)
        fixed_text_str = melt.first_nrows(fixed_text_str, FLAGS.num_fixed_evaluate_examples)

        #notice read data always be FLAGS.fixed_eval_batch_size, if only 5 tests then will wrapp the data 
        eval_image_name = tf.concat([fixed_image_name, eval_image_name], 0)
        eval_image_feature = tf.concat([fixed_image_feature, eval_image_feature], 0)

        #melt.align only need if you use dynamic batch/padding
        if FLAGS.dynamic_batch_length:
          fixed_text, eval_text = melt.align_col_padding2d(fixed_text, eval_text)
        eval_text = tf.concat([fixed_text, eval_text], 0)
        eval_text_str = tf.concat([fixed_text_str, eval_text_str], 0)
        eval_batch_size = FLAGS.num_fixed_evaluate_examples + FLAGS.eval_batch_size 
      
      #should aways be FLAGS.num_fixed_evaluate_examples + FLAGS.num_evaluate_examples
      num_evaluate_examples = min(eval_batch_size, FLAGS.num_fixed_evaluate_examples + FLAGS.num_evaluate_examples)

      #------------combine valid and fixed valid 
      evaluate_image_name = melt.first_nrows(eval_image_name, num_evaluate_examples)
      evaluate_image_feature = melt.first_nrows(eval_image_feature, num_evaluate_examples)
      evaluate_text_str = melt.first_nrows(eval_text_str, num_evaluate_examples)
      evaluate_text = melt.first_nrows(eval_text, num_evaluate_examples)

      self.num_evaluate_examples = num_evaluate_examples
      
      eval_result = eval_image_name, eval_image_feature, eval_text, eval_text_str
      eval_show_result = evaluate_image_name, evaluate_image_feature, evaluate_text, evaluate_text_str
    
    return eval_result, eval_show_result, eval_batch_size

  def gen_valid_neg_input(self, inputs, decode_neg, trainset, eval_batch_size):
    eval_neg_text, eval_neg_text_str = inputs(
      trainset, 
      decode=decode_neg,
      batch_size=eval_batch_size * FLAGS.num_negs,
      num_epochs=0, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.eval_shuffle,
      fix_random=FLAGS.eval_fix_random,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      fix_sequence=FLAGS.fix_sequence,
      name=self.input_valid_neg_name)

    if FLAGS.feed_dict:
      self.eval_neg_text_str = eval_neg_text_str
      eval_neg_text = self.neg_text_place

      if FLAGS.show_eval:
        self.eval_neg_text_str_place = text_str_placeholder('eval_neg_text_str_place')
        self.eval_neg_text_str = eval_neg_text_str
        eval_neg_text_str = self.eval_neg_text_str_place
      
    eval_neg_text, eval_neg_text_str = input.reshape_neg_tensors([eval_neg_text, eval_neg_text_str], 
                                                            eval_batch_size, 
                                                            FLAGS.num_negs)
    #[batch_size, num_negs, 1] -> [batch_size, num_negs], notice tf.squeeze will get [batch_size] when num_negs == 1
    eval_neg_text_str = tf.squeeze(eval_neg_text_str, squeeze_dims=[-1])

    return eval_neg_text, eval_neg_text_str

  def gen_input(self, train_only=False):
    timer = gezi.Timer('gen input')

    assert not (FLAGS.feed_dict and FLAGS.dynamic_batch_length), \
          'if use feed dict then must use fixed batch length, or use buket mode(@TODO)'

    input_results = {}

    input_name_list = [self.input_train_name, self.input_train_neg_name, \
                       self.input_valid_name, self.fixed_input_valid_name, \
                       self.input_valid_neg_name]

    #for name in input_name_list:
    #  input_results[name] = None

    inputs, decode, decode_neg = input.get_decodes(FLAGS.shuffle_then_decode, FLAGS.dynamic_batch_length, use_neg=(FLAGS.num_negs > 0))

    input_results[self.input_train_name], trainset = self.gen_train_input(inputs, decode)

    print('decode_neg', decode_neg)

    if decode_neg is not None:
      input_results[self.input_train_neg_name] = self.gen_train_neg_input(inputs, decode_neg, trainset)
    
    if not train_only:
      #---------------------- valid
      train_with_validation = bool(FLAGS.valid_input) 
      self.train_with_validation = train_with_validation
      print('train_with_validation:', train_with_validation)
      if train_with_validation:
        input_results[self.input_valid_name], \
        input_results[self.fixed_input_valid_name], \
        eval_batch_size = self.gen_valid_input(inputs, decode)

        if decode_neg is not None:
          input_results[self.input_valid_neg_name] = self.gen_valid_neg_input(inputs, decode_neg, trainset, eval_batch_size)

    print_input_results(input_results)

    timer.print()

    return input_results
