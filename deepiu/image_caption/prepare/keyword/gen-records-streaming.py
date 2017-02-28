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

flags.DEFINE_string('output_directory', './output', '')
flags.DEFINE_integer('part', 0, '')
flags.DEFINE_integer('mode', 0, '0 gen bin and gen count, 1 gen bin only, 2 gen count only')
flags.DEFINE_string('vocab', './vocab.bin', 'vocabulary binary file')
flags.DEFINE_boolean('pad', False, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('input', '/home/gezi/data/image-text-sim/train/train', 'input pattern')
flags.DEFINE_string('name', 'train', '')
#flags.DEFINE_string('seg_method', 'basic', '')

"""
use only top query?
use all queries ?
use all queries but deal differently? add weight to top query?
"""

import sys,os
import numpy as np
import melt
import gezi

#import libgezi


import conf  
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

#cp conf.py ../../../ before
from deepiu.image_caption import text2ids

#segmentor = gezi.Segmentor()

#from libword_counter import Vocabulary 
#vocabulary = Vocabulary(FLAGS.vocab, NUM_RESERVED_IDS)

#print('vocab:', FLAGS.vocab, file=sys.stderr)
#assert vocabulary.size() > NUM_RESERVED_IDS
#print('vocab size:', vocabulary.size(), file=sys.stderr)

print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
assert ENCODE_UNK == text2ids.ENCODE_UNK
print('seg_method', FLAGS.seg_method, file=sys.stderr)

text2ids.init()

writer = None
if FLAGS.mode != 2:
  gezi.try_mkdir(FLAGS.output_directory)

  outfile = '%s/%s_%s'%(FLAGS.output_directory, FLAGS.name, FLAGS.part)
  print('outfile:', outfile, file=sys.stderr) 
  
  writer = melt.tfrecords.Writer(outfile)

num = 0
count = 0
for line in sys.stdin:
  if num % 1000 == 0:
   print(num, file=sys.stderr)
  num += 1
  l = line.rstrip().split('\t')
  img = l[0]
  img_end = IMAGE_FEATURE_LEN + 1
  img_feature = [float(x) for x in l[1: img_end]]
  texts = [x.split('\x01')[0] for x in l[img_end:]]
  for text in texts:
    if text.strip() == '':
      continue
    word_ids = text2ids.text2ids(text, seg_method=FLAGS.seg_method, feed_single=FLAGS.feed_single, allow_all_zero=True, pad=False)
    word_ids_length = len(word_ids)
    if num % 1000 == 0:
     #print(libgezi.gbk2utf8('\t'.join(words)), file=sys.stderr)
     #print('\t'.join(words), file=sys.stderr)
     print(text, word_ids, text2ids.ids2text(word_ids), file=sys.stderr)
    if len(word_ids) == 0:
      continue
    word_ids = word_ids[:TEXT_MAX_WORDS]
    if FLAGS.pad:
      word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS, 0)
    
    if writer is not None:
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_name': melt.bytes_feature(img),
        'image_feature': melt.float_feature(img_feature),
        'text': melt.int_feature(word_ids),
        'text_str': melt.bytes_feature(text),
        }))
      writer.write(example)
    else:
      count += 1

if FLAGS.mode != 1:
  if writer is not None:
    count = writer.count
  print('count\t%d'%(count), file=sys.stderr)
  #--------for calc total count
  print('count\t%d'%(count))

if writer is not None:
  writer.close()
