#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2016-08-19 01:31:54.834381
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import melt
from melt.models import Mlp

from melt import layers

#this is input data related here is just demo usage,for our data has 34 classes and 324510 features
NUM_CLASSES = 2
NUM_FEATURES = 488

def build_graph(X, y):
  #algo = Mlp(input_dim=NUM_FEATURES, num_classes=NUM_CLASSES)
  #py_x = algo.forward(X)

  hidden_size = 200
  py_x = layers.mlp(X, hidden_size, NUM_CLASSES)
  
  loss = melt.sparse_softmax_cross_entropy(py_x, y)
  tf.scalar_summary('loss_%s'%loss.name, loss)
 
  #tf.scalar_summary('loss', loss)
  ##loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(py_x, y))

  accuracy = melt.precision_at_k(py_x, y, 1)
  tf.scalar_summary('precision@1_%s'%accuracy.name, accuracy)

  #below will cause  tensorflow.python.framework.errors.InvalidArgumentError: Duplicate tag precicsion@1 found in summary inputs
  #the problem here is we want to share all other things but without scalar summarys in graph
  #so if we want to build more than once ... then scalar_summary must use op.name
  #or else do it outof build_graph, setting names by yourself!
  #eval_loss, eval_accuracy = build_grapy(X, y)
  #tf.scalar_summary('eval_loss', eval_loss) 
  #tf.scalar_summary('eval_accuracy', eval_accuracy)
  #since tensorboard has 'Split on underscores', so for better comparaion side by side
  #loss_train, loss_eval, accuracy_train, accuracy_eval is better then train_loss,eval_loss 

  #tf.scalar_summary('precicsion@1', accuracy)
  return loss, accuracy
