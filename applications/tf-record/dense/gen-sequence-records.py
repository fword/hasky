#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-sequence-records.py
#        \author   chenghuige  
#          \date   2016-09-11 11:45:47.140270
#   \Description  
# ==============================================================================

"""
This is file is for experimenting purpose for dense fixed len dataset like urate,
use gen-recordes.py using Example will be ok
here will show the usage for var len sequence dataset used for lstm or other seq method,
but we will use urate dataset for test, and make fake var len for some examples,
so we can test padding

the benifit here is we will not pad and save space, tf will then pad for us

var len and dense !
see spares/read-reocords.py for sparse example reading,  sparse seq example reading @TODO
will not need spares at most times, only need sparse right now fow cbow text classification, see examples/sparse-tensor-classification
mostly used will be padding to dense

@TODO how to better handle one image mulitple text(click query) (without having to use sparse?) ? Now will waste space store images..

http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/#more-820
tensorflow has built-in support for batch padding. If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.

But now only batch support dynamic_pad, batch_join and shuffle_batch not..

@TODO consider tf.nn.sparse_ops.sparse_to_dense

python gen-sequence-records.py /home/gezi/data/urate/train /tmp/urate.train 
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_examples', 100, 'Batch size.')
flags.DEFINE_boolean('fake_var_len', False, 'for testing popurse make some examples fake len, to make length unequal')

import melt

def main(argv):
  writer = tf.python_io.TFRecordWriter(argv[2])
  num = 0
  for line in open(argv[1]):
    if line[0] == '#':
      continue
    if num % 10000 == 0:
      print('%d lines done'%num)
    l = line.rstrip().split()
    
    label_index = 0
    if l[0][0] == '_':
      label_index = 1
      id = int(l[0][1:])
    else:
      id = num
    label = int(l[label_index])
    
    start = label_index + 1
    feature = [float(x) for x in l[start:]]

    if FLAGS.fake_var_len:
      if id % 2 == 0:
        feature = feature[:10]

      if id % 3 == 0:
        feature = feature[:20]

    example =  tf.train.SequenceExample(
      context=melt.features(
        {
        'id': melt.int_feature(id), 
        'label': melt.int_feature(label)
        }),
      feature_lists=melt.feature_lists(
        { 
          #see sequence_test.py use each single as a list and stack all lists(single items)
          #can this deal with var len sequence ?
          'feature': melt.feature_list([melt.float_feature(item) for item in feature])
          #'feature': melt.feature_list(melt.float_feature(feature))
        }))
    
    writer.write(example.SerializeToString())
    
    num += 1
    if FLAGS.num_examples and num == FLAGS.num_examples:
      break

if __name__ == '__main__':
  tf.app.run()
