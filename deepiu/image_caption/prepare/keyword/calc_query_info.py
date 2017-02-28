#!/usr/bin/env python
# ==============================================================================
#          \file   calc_query_info.py
#        \author   chenghuige  
#          \date   2016-08-25 23:55:21.934224
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

import conf
from conf import IMAGE_FEATURE_LEN

num_words = 0
num = 0
for line in open(sys.argv[1]):
  l = line.split('\t')
  img_end = IMAGE_FEATURE_LEN + 1
  words = l[img_end:]
  words = [x.split('\x01')[0] for x in words]
  num_words += len(words)
  #print '\t'.join(keywords)
  num += 1
  if num == 100000:
    break

print num_words / num
