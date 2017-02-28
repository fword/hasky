#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   count.py
#        \author   chenghuige  
#          \date   2016-10-08 23:39:45.378793
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

count = 0
num_lines = 0
for line in sys.stdin:
  count += int(line.rstrip().split('\t')[-1])
  num_lines += 1
 
print('%d\t%d' % (count, num_lines))
