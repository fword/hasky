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

"""
maily used for small dataset(user set fixed evaluation set for show)
"""
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')
flags.DEFINE_boolean('pad', True, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_directory', '/tmp/valid/',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input', '/home/gezi/data/image-text-sim/evaluate/result.txt', 'input pattern')
flags.DEFINE_string('name', 'test', '')

import sys,os

import numpy as np
import melt
import gezi

import nowarning
from libsegment import *
#need ./data ./conf
Segmentor.Init()

from libword_counter import Vocabulary 
vocabulary = Vocabulary(FLAGS.vocab)

TEXT_MAX_WORDS = 10

texts = []
text_strs = []


def deal_file(file, writer):
  num = 0
  for line in open(file):
    if num % 1000 == 0:
      print('num:', num)
    l = line.rstrip().split('\t')
    img = l[0]
    img_feature = [float(x) for x in l[1:1001]]
    text = l[-1].split('\x01')[0]
    words = Segmentor.Segment(text)
    word_ids = [vocabulary.id(word) for word in words if vocabulary.has(word)]
    if len(word_ids) == 0:
      num += 1
      continue
    if FLAGS.pad:
      word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS, 0)
    
    texts.append(word_ids)
    text_strs.append(text)
    example = tf.train.Example(features=tf.train.Features(feature={
      'image_name': melt.bytes_feature(img),
       'image_feature': melt.float_feature(img_feature),
       'text': melt.int_feature(word_ids),
       'text_str': melt.bytes_feature(text),
       }))
    #writer.write(example.SerializeToString())
    writer.write(example)
    num += 1
  
  #texts_dict[thread_index] = gtexts[thread_index]
  #text_strs_dict[thread_index] = gtext_strs[thread_index]


#writer = tf.python_io.TFRecordWriter('{}/{}'.format(FLAGS.output_directory, FLAGS.name))
with melt.tfrecords.Writer('{}/{}'.format(FLAGS.output_directory, FLAGS.name)) as writer:
  deal_file(FLAGS.input, writer)

#texts = [val for sublist in texts_dict.values() for val in sublist]
#text_strs = [val for sublist in text_strs_dict.values() for val in sublist]

print('len(texts):', len(texts))
np.save(os.path.join(FLAGS.output_directory, 'texts.npy'), np.array(texts))
np.save(os.path.join(FLAGS.output_directory, 'text_strs.npy'), np.array(text_strs))
