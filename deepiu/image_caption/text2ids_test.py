#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test_text2ids.py
#        \author   chenghuige  
#          \date   2016-09-05 15:27:35.019551
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import text2ids

ids = text2ids.text2ids('Oh q the same thing to me', 'phrase_single', feed_single=True)
print(ids)

ids = text2ids.text2ids('Oh q the same thing to me')
print(ids)