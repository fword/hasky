#!/usr/bin/env python
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description  
# ==============================================================================
"""
using pool ctrl + c will not stop...
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')
flags.DEFINE_boolean('pad', True, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_directory', '/tmp/',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input', '/home/gezi/data/image-text-sim/test/test_*', 'input pattern')
flags.DEFINE_string('name', 'train', '')
flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

"""
use only top query?
use all queries ?
use all queries but deal differently? add weight to top query?
"""

import sys,os
import multiprocessing
from multiprocessing import Process, Manager, Value

#import glob
import gezi
glob = gezi.bigdata_util

import numpy as np
import melt
import gezi

import nowarning
#from libsegment import *
#need ./data ./conf
#Segmentor.Init()
Segmentor = gezi.Segmentor()

from libword_counter import Vocabulary 
vocabulary = Vocabulary(FLAGS.vocab)

import conf  
from conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN

texts = []
text_strs = []

manager = Manager()
texts_dict = manager.dict()
text_strs_dict = manager.dict()
gtexts = [[]] * FLAGS.threads
gtext_strs = [[]] * FLAGS.threads

def deal_file(thread_index, file):
  out_file = '{}/{}_{}'.format(FLAGS.output_directory, FLAGS.name, thread_index) if FLAGS.threads > 1 else '{}/{}'.format(FLAGS.output_directory, FLAGS.name)
  writer = melt.tfrecords.Writer(out_file)
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
      words = Segmentor.Segment(text)
      word_ids = [vocabulary.id(word) for word in words if vocabulary.has(word)]
      if len(word_ids) == 0:
        continue
      if FLAGS.pad:
        word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS, 0)
    
      gtexts[thread_index].append(word_ids)
      gtext_strs[thread_index].append(text)

      #add pos info? weght info? or @TODO add click num info
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_name': melt.bytes_feature(img),
        'image_feature': melt.float_feature(img_feature),
        'text': melt.int_feature(word_ids),
        'text_str': melt.bytes_feature(text),
        }))
      writer.write(example)
    num += 1
  
  texts_dict[thread_index] = gtexts[thread_index]
  text_strs_dict[thread_index] = gtext_strs[thread_index]

files = glob.glob(FLAGS.input)
print(FLAGS.input, files, len(files))
pool = multiprocessing.Pool(processes = FLAGS.threads)

#i, file NOTICE input files sequence may be _1 _0 but output to _0, _1..
for args in enumerate(files):
  pool.apply_async(deal_file, args)

pool.close()
pool.join()

texts = [val for sublist in texts_dict.values() for val in sublist]
text_strs = [val for sublist in text_strs_dict.values() for val in sublist]

print('len(texts):', len(texts))
np.save(os.path.join(FLAGS.output_directory, 'texts.npy'), np.array(texts))
np.save(os.path.join(FLAGS.output_directory, 'text_strs.npy'), np.array(text_strs))
