#!/usr/bin/env python
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2016-07-18 21:25:57.445679
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('image_feature', '/home/gezi/data/image-auto-comment/train/img2fea.txt', 'input image feature file')
flags.DEFINE_string('text', '/home/gezi/data/image-auto-comment/train/results_20130124.token', 'input image text file')
flags.DEFINE_string('vocab', '/tmp/train.comment/vocab.bin', 'vocabulary binary file')

flags.DEFINE_string('output_directory', '/tmp/train/',
                         """Directory to download data files and write the 
                         converted result""")

flags.DEFINE_integer('shards', 12, 'Number of shards in TFRecord files.')

flags.DEFINE_string('name', 'train', 'records name')

flags.DEFINE_boolean('pad', False, 'wether to pad to pad 0 to make fixed length text ids')

flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

flags.DEFINE_integer('num_records', 0, '')

flags.DEFINE_boolean('np_save', True, 'np save text ids and text')

flags.DEFINE_boolean('debug', False, 'debug mode for test')

flags.DEFINE_integer('ori_text_index', -2, """defualt assume the last colums as ori_text, 
                                              text(segged text) so -2, for en corpuse wich do not 
                                              need seg, will only have ori_text so ori_text index will be -1 """)

import os

import threading
import multiprocessing
from multiprocessing import Process, Manager, Value

import numpy as np

import gezi
from libprogress_bar import ProgressBar
from libword_counter import Vocabulary 

import melt

import conf  
from conf import WORDS_SEP, TEXT_MAX_WORDS

NUM_DEBUG_LINES = 1000

vocabulary = Vocabulary(FLAGS.vocab)

text_map = {}
text_map_ = {}

#how many records generated
counter = Value('i', 0)
#the max num words of the longest text
max_num_words = Value('i', 0)
#the total words of all text
sum_words = Value('i', 0)

manager = Manager()
texts_dict = manager.dict()
text_strs_dict = manager.dict()
gtexts = [[]] * FLAGS.threads
gtext_strs = [[]] * FLAGS.threads

#--------- same text for the same image will treat as only 1
def parse_text_file(text_file):
  num_lines = gezi.get_num_lines(text_file)
  pb = ProgressBar(num_lines, 'parse text file %s'%text_file)
  for line in open(text_file):
    pb.progress()
    l = line.split('\t')
    image = l[0]
    image = image[:image.index('#')]
    text = l[-1].strip()
    #why text and ori_text ? because fo cn corpus text will be \x01 seperated(segged text)
    #for en corpus text and ori_text is the same
    ori_text = l[FLAGS.ori_text_index].strip() 
    if text == '':
      continue
    if image not in text_map:
      text_map_[image] = set([text])
      text_map[image] = [(text, ori_text)]
    else:
      if text not in text_map_:
        text_map_[image].add(text)
        text_map[image].append((text, ori_text))
  for image in text_map:
    text_map[image] = list(text_map[image])

images = {}
def _parse_line(line, writer, thread_index = 0):
  l = line.rstrip().split('\t')
  image_name = l[0]
  image_feature = [float(x) for x in l[1:]]
  if image_name not in text_map:
    print('image %s ignore'%image_name)
    return
  else:
    for text, ori_text in text_map[image_name]:
      word_ids = [vocabulary.id(word) for word in text.split(WORDS_SEP) if vocabulary.has(word)]
      if not word_ids:
        continue 
      word_ids_length = len(word_ids)
      word_ids = word_ids[:TEXT_MAX_WORDS]
      if FLAGS.pad:
        word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS)
     
      if FLAGS.np_save:
        gtexts[thread_index].append(word_ids)
        gtext_strs[thread_index].append(ori_text)
 
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_name': melt.bytes_feature(image_name),
        'image_feature': melt.float_feature(image_feature),
        'text': melt.int_feature(word_ids),
        'text_str': melt.bytes_feature(ori_text),
        }))
      
      #NOTICE not test here for num_threads > 1
      if FLAGS.num_records:
        if image_name not in images:
          images[image_name] = 1
          print(image_name, len(images))
          writer.write(example.SerializeToString())
          if len(images) == FLAGS.num_records:
            print('Done')
            exit(1)
      else:
        writer.write(example.SerializeToString())
        global counter, max_num_words, sum_words
        with counter.get_lock():
          counter.value += 1
        if word_ids_length > max_num_words.value:
          with max_num_words.get_lock():
            max_num_words.value = word_ids_length
        with sum_words.get_lock():
          sum_words.value += word_ids_length
        
def _convert_to(f, name, thread_index, start, end):
  num_shards = FLAGS.shards
  shard = thread_index
  output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
  output_file = os.path.join(FLAGS.output_directory, output_filename)
  print('Writing', output_file, start)
  writer = tf.python_io.TFRecordWriter(output_file)
  num_lines = end - start
  pb = ProgressBar(num_lines)
  for i in xrange(start, end):
  	pb.progress()
  	line = f[i]
  	_parse_line(line, writer, thread_index)
  
  texts_dict[thread_index] = gtexts[thread_index]
  text_strs_dict[thread_index] = gtext_strs[thread_index]

def convert_to(feat_file, name):
  num_shards = FLAGS.shards
  num_threads = FLAGS.threads
  if FLAGS.threads > 1:
    assert(num_threads == num_shards)
    f = open(feat_file).readlines()
    num_lines = len(f)
    if FLAGS.debug:
      num_lines = NUM_DEBUG_LINES
    shard_ranges = np.linspace(0,
                             num_lines,
                             num_shards + 1).astype(int)
    
    record = []
    for i in xrange(num_threads):
      args = (f, name, i, shard_ranges[i], shard_ranges[i + 1])
      process = multiprocessing.Process(target=_convert_to,args=args)
      process.start()
      record.append(process)

    for process in record:
      process.join()
    return

  #--------------single thread
  num_lines = gezi.get_num_lines(feat_file)
  if FLAGS.debug:
    num_lines = NUM_DEBUG_LINES
  shard_ranges = np.linspace(0,
                             num_lines,
                             num_shards + 1).astype(int)
  pb = ProgressBar(num_lines, "convert")
  shard = 0
  count = 0;
  
  output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
  output_file = os.path.join(FLAGS.output_directory, output_filename)
  print('Writing', output_file, count)
  writer = tf.python_io.TFRecordWriter(output_file)
 
  for line in open(feat_file):
    pb.progress()
    if count >= shard_ranges[shard + 1]:
      shard += 1
      writer.close()
      output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
      output_file = os.path.join(FLAGS.output_directory, output_filename)
      print('Writing', output_file, count)
      writer = tf.python_io.TFRecordWriter(output_file)
    _parse_line(line, writer)

    count += 1
    if FLAGS.debug and count >= NUM_DEBUG_LINES:
      break
   
  writer.close()
  

def main(argv):
  parse_text_file(FLAGS.text)
  gezi.try_mkdir(FLAGS.output_directory)
  convert_to(FLAGS.image_feature, FLAGS.name)

  if FLAGS.np_save:
    if FLAGS.threads == 1:
      global gtexts, gtext_strs
      texts = [val for sublist in gtexts for val in sublist]
      text_strs = [val for sublist in gtext_strs for val in sublist]
    else:
      texts = [val for sublist in texts_dict.values() for val in sublist]
      text_strs = [val for sublist in text_strs_dict.values() for val in sublist]

    print('len(texts):', len(texts))
    np.save(os.path.join(FLAGS.output_directory, 'texts.npy'), np.array(texts))
    np.save(os.path.join(FLAGS.output_directory, 'text_strs.npy'), np.array(text_strs))
  
  num_records = counter.value
  print('num_records:', num_records)
  gezi.write_to_txt(num_records, os.path.join(FLAGS.output_directory, 'num_records.txt'))

  print('counter:', counter.value)
  print('max_num_words:', max_num_words.value)
  print('avg_num_words:', sum_words.value / counter.value)

if __name__ == '__main__':
  tf.app.run()
