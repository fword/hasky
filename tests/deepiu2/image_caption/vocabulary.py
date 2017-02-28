#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   vocabulary.py
#        \author   chenghuige  
#          \date   2016-10-06 21:57:19.276673
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import deepiu2
print(dir(deepiu2))

#@NOTICE use from not import!
#from deepiu2.image_caption import conf
import deepiu2.image_caption.conf as conf
