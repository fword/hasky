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
from melt.models import mlp


g_num_features, g_num_classes =  -1, -1

def set_input_info(num_features, num_classes):
  global g_num_features, g_num_classes
  g_num_features = num_features
  g_num_classes = num_classes

def predict(X):
  return mlp.forward(X, 
                    input_dim=g_num_features, 
                    num_outputs=g_num_classes, 
                    hiddens=[200,100,50])
                    #hiddens=[200])

def build_graph(X, y):
  #---build forward graph
  py_x = predict(X)
  
  #-----------for classification we can set loss function and evaluation metrics,so only forward graph change
  #---set loss function
  loss = melt.sparse_softmax_cross_entropy(py_x, y)

  #---choose evaluation metrics
  accuracy = melt.precision_at_k(py_x, y, 1)

  return loss, accuracy
