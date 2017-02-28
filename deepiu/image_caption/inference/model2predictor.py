#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model2predictor.py
#        \author   chenghuige  
#          \date   2016-09-30 15:19:43.440970
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '../model.flickr.show_and_tell2/', '')
flags.DEFINE_string('meta_graph', '', '')

flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('algo', 'show_and_tell', '')

flags.DEFINE_integer('seq_decode_method', 0, 'sequence decode method: 0 max prob, 1 sample, 2 full sample, 3 beam search')
flags.DEFINE_integer('beam_size', 5, 'for seq decode beam search size')
#flags.DEFINE_boolean('ignore_unk', True, '')

flags.DEFINE_integer('num_images', 1000, '')
flags.DEFINE_integer('batch_size', 1000, '')

flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                                                set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')

  ##---for predict without creating predict graph from scratch(load from checkpoint meta data)
  #if predictor is not None:
  #  if FLAGS.algo == Algos.show_and_tell:
  #    predictor.init_predict_text(decode_method=FLAGS.seq_decode_method, 
  #                                beam_size=FLAGS.beam_size,
  #                                ignore_unk=False)
  #    predictor.init_predict_text(decode_method=FLAGS.seq_decode_method, 
  #                                beam_size=FLAGS.beam_size,
  #                               ignore_unk=True)
  #    for item in predictor.text_list:
  #      tf.add_to_collection('text', item)

import sys 
sys.path.append('../')

saver = tf.train.Saver()
saver.export_meta_graph
import melt

def convert(model_dir, meta_graph):
  model_dir, model_path = melt.get_model_dir_and_path(model_dir)
  if not meta_graph:
    meta_graph = '%s/graph.meta'%model_dir



def main(_):
  convert(FLAGS.model_dir, FLAGS.meta_graph)

if __name__ == '__main__':
  tf.app.run()