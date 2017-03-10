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

flags.DEFINE_string('input', '/home/gezi/data/textsum', 'input pattern')
flags.DEFINE_string('name', 'train', '')
flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

flags.DEFINE_boolean('np_save', True, 'np save text ids and text')

flags.DEFINE_integer('max_lines', 0, '')

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
from deepiu.util import text2ids

import conf  
from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK


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


def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, seg_method=FLAGS.seg_method, feed_single=FLAGS.feed_single, allow_all_zero=True, pad=False)
  word_ids_length = len(word_ids)

  if len(word_ids) == 0:
    return []
  word_ids = word_ids[:max_words]
  if FLAGS.pad:
    word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def deal_file(file, thread_index):
  out_file = '{}/{}_{}'.format(FLAGS.output_directory, FLAGS.name, thread_index) if FLAGS.threads > 1 else '{}/{}'.format(FLAGS.output_directory, FLAGS.name)
  print('out_file:', out_file)
  with melt.tfrecords.Writer(out_file) as writer:
    num = 0
    for line in open(file):
      line = line.lower()
      if num % 1000 == 0:
        print(num)
      if FLAGS.max_lines and num >= FLAGS.max_lines:
        break
      l = line.rstrip().split('\t')
      #@TODO from text -> ids should move out so online code can share it for evaluation or use for feed dict
      #words = segmentor.Segment(text, FLAGS.seg_method)
      #word_ids = [vocabulary.id(word) for word in words if vocabulary.has(word) or ENCODE_UNK]

      #text is what to predict which is clickquery right now  decoder
      #input text is what to predict from, encoder, here will may be ct0, title, real_title

      clickquery = l[-4]
      ct0 = l[-3]
      title = l[-2]
      real_title = l[-1]

      if title.strip() is '':
        title = real_title

      if clickquery.startswith('http://'):
        clickquery = l[3]

      text = clickquery
      word_ids = _text2ids(text, TEXT_MAX_WORDS)

      if not word_ids:
        continue
      
      if FLAGS.np_save:
        gtexts[thread_index].append(word_ids)
        gtext_strs[thread_index].append(text)
      

      ct0_ids = _text2ids(ct0, INPUT_TEXT_MAX_WORDS)


      title_ids = _text2ids(title, INPUT_TEXT_MAX_WORDS)
      real_title_ids = _text2ids(real_title, INPUT_TEXT_MAX_WORDS)
      
      if len(ct0_ids) == 0:
        ct0_ids = real_title_ids 
        ct0 = real_title
      
      if num % 1000 == 0:
        print(text, word_ids, text2ids.ids2text(word_ids), file=sys.stderr)
        print(ct0, ct0_ids, text2ids.ids2text(ct0_ids), file=sys.stderr)

      image = l[1]
      url = l[2]
      
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_name': melt.bytes_feature(image),
        'url': melt.bytes_feature(url),
        'text_str': melt.bytes_feature(text),
        'ct0_str': melt.bytes_feature(ct0),
        'title_str': melt.bytes_feature(title),
        'real_title_str': melt.bytes_feature(real_title),
        'text': melt.int_feature(word_ids),
        'ct0': melt.int_feature(ct0_ids),
        'title': melt.int_feature(title_ids),
        'real_title': melt.int_feature(real_title_ids),
        }))
      writer.write(example)
      
      global counter, max_num_words, sum_words
      with counter.get_lock():
        counter.value += 1
      word_ids_length = len(word_ids)
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
