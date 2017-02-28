#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_flags.py
#        \author   chenghuige  
#          \date   2016-12-24 17:02:28.330058
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
 
flags.DEFINE_string('cell', 'lstm_block', 'now lstm or gru?? lstm_block is fastest')
flags.DEFINE_boolean('bidirectional', False, 'wether use bidirectional rnn')
flags.DEFINE_float('keep_prob', 1, 'or 0.9 0.8 0.5')
flags.DEFINE_integer('num_layers', 1, 'or > 1')
flags.DEFINE_boolean('feed_initial_sate', False, """set true just like ptb_word_lm to feed 
                                                  last batch final state to be inital state 
                                                  but experiments not show better result(similar)""")