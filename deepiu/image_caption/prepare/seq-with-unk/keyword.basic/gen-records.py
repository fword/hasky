#!/usr/bin/env python
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')
flags.DEFINE_boolean('pad', False, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_directory', '/tmp/train/',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input', '/home/gezi/data/image-text-sim/train/train', 'input pattern')
flags.DEFINE_string('name', 'train', '')
flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

flags.DEFINE_boolean('np_save', True, 'np save text ids and text')
#flags.DEFINE_string('seg_method', 'default', '')

"""
use only top query?
use all queries ?
use all queries but deal differently? add weight to top query?
"""

import sys,os
import multiprocessing
from multiprocessing import Process, Manager, Value

import numpy as np
import melt
import gezi

#cp conf.py ../../../ before
from deepiu.image_caption import text2ids

import conf  
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK


print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
assert ENCODE_UNK == text2ids.ENCODE_UNK

#from libsegment import *
#need ./data ./conf
#Segmentor.Init()
#segmentor = gezi.Segmentor()

#from libword_counter import Vocabulary 
#vocabulary = Vocabulary(FLAGS.vocab, NUM_RESERVED_IDS)

texts = []
text_strs = []

manager = Manager()
texts_dict = manager.dict()
text_strs_dict = manager.dict()
gtexts = [[]] * FLAGS.threads
gtext_strs = [[]] * FLAGS.threads

#how many records generated
counter = Value('i', 0)
#the max num words of the longest text
max_num_words = Value('i', 0)
#the total words of all text
sum_words = Value('i', 0)

text2ids.init()

def deal_file(file, thread_index):
  out_file = '{}/{}_{}'.format(FLAGS.output_directory, FLAGS.name, thread_index) if FLAGS.threads > 1 else '{}/{}'.format(FLAGS.output_directory, FLAGS.name)
  print('out_file:', out_file)
  with melt.tfrecords.Writer(out_file) as writer:
    num = 0
    for line in open(file):
      if num % 1000 == 0:
        print(num)
      l = line.rstrip().split('\t')
      img = l[0]
      img_end = IMAGE_FEATURE_LEN + 1
      img_feature = [float(x) for x in l[1: img_end]]
      texts = [x.split('\x01')[0] for x in l[img_end:]]
      for text in texts:
        #@TODO from text -> ids should move out so online code can share it for evaluation or use for feed dict
        #words = segmentor.Segment(text, FLAGS.seg_method)
        #word_ids = [vocabulary.id(word) for word in words if vocabulary.has(word) or ENCODE_UNK]
        word_ids = text2ids.text2ids(text, seg_method=FLAGS.seg_method, feed_single=FLAGS.feed_single, allow_all_zero=True, pad=False)
        word_ids_length = len(word_ids)
        if num % 1000 == 0:
          print(text, word_ids, file=sys.stderr)
        if len(word_ids) == 0:
          continue
        word_ids = word_ids[:TEXT_MAX_WORDS]
        if FLAGS.pad:
          word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS, 0)
      
        if FLAGS.np_save:
          gtexts[thread_index].append(word_ids)
          gtext_strs[thread_index].append(text)

        assert img and img_feature and word_ids and text, line
        assert len(img_feature) == IMAGE_FEATURE_LEN
        #add pos info? weght info? or @TODO add click num info
        example = tf.train.Example(features=tf.train.Features(feature={
          'image_name': melt.bytes_feature(img),
          'image_feature': melt.float_feature(img_feature),
          'text': melt.int_feature(word_ids),
          'text_str': melt.bytes_feature(text),
          }))
        writer.write(example)
        
        global counter, max_num_words, sum_words
        with counter.get_lock():
          counter.value += 1
        if word_ids_length > max_num_words.value:
          with max_num_words.get_lock():
            max_num_words.value = word_ids_length
        with sum_words.get_lock():
          sum_words.value += word_ids_length
      num += 1
  
  texts_dict[thread_index] = gtexts[thread_index]
  text_strs_dict[thread_index] = gtext_strs[thread_index]


record = []
for thread_index in xrange(FLAGS.threads):
  in_file = '{}_{}'.format(FLAGS.input, thread_index) if FLAGS.threads > 1 else FLAGS.input
  args = (in_file,thread_index)
  process = multiprocessing.Process(target=deal_file,args=args)
  process.start()
  record.append(process)

for process in record:
  process.join()

if FLAGS.np_save:
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
