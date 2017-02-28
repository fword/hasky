#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   test_seg.py
#        \author   chenghuige  
#          \date   2016-09-05 11:48:05.006754
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gezi
import libsegment

seg = gezi.Segmentor()

print('\t'.join(seg.Segment('美女一定要支持')))
print('\x01'.join(seg.Segment('Oh q the same thing to me')))
print('\x01'.join(seg.Segment('Oh q the same thing to me', 'phrase_single')))
print('\x01'.join(seg.Segment('Oh q the same thing to me', 'phrase')))
print('\t'.join(seg.Segment('绿鹭')))
print('\t'.join(seg.segment('绿鹭')))
print('\t'.join(seg.segment_phrase('绿鹭')))
print('\t'.join(gezi.seg.Segment('绿鹭', libsegment.SEG_NEWWORD)))
print('\t'.join(gezi.seg.Segment('绿鹭')))

print('|'.join(gezi.segment_char('a baby is looking at 我的小伙伴oh 我不no no没关系 是不是   tian, that not ')))


from libword_counter import Vocabulary

v = Vocabulary('/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 2)
print(v.id('美女'))
print(v.key(v.id('美女')))
