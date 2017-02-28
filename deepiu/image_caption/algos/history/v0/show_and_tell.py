#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
@TODO
set to cpu but still outof memory  vocabulary size 50w
sh train-keyword-lstm.sh   
W tensorflow/core/common_runtime/bfc_allocator.cc:271] Ran out of memory trying to allocate 488.28MiB.  See logs for memory state.
W tensorflow/core/framework/op_kernel.cc:936] Resource exhausted: OOM when allocating tensor with shape[256,500001]
E tensorflow/core/client/tensor_c_api.cc:485] OOM when allocating tensor with shape[256,500001]
   [[Node: gradients/AddN_3 = AddN[N=21, T=DT_FLOAT, _class=["loc:@gradients/RNN/MatMul_20_grad/MatMul_1"], _device="/job:localhost/replica:0/task:0/gpu:0"](gradients/RNN/MatMul_20_grad/tuple/control_dependency_1, gradients/RNN/MatMul_19_grad/tuple/control_dependency_1, gradients/RNN/MatMul_18_grad/tuple/control_dependency_1, gradients/RNN/MatMul_17_grad/tuple/control_dependency_1, gradients/RNN/MatMul_16_grad/tuple/control_dependency_1, gradients/RNN/MatMul_15_grad/tuple/control_dependency_1, gradients/RNN/MatMul_14_grad/tuple/control_dependency_1, gradients/RNN/MatMul_13_grad/tuple/control_dependency_1, gradients/RNN/MatMul_12_grad/tuple/control_dependency_1, gradients/RNN/MatMul_11_grad/tuple/control_dependency_1, gradients/RNN/MatMul_10_grad/tuple/control_dependency_1, gradients/RNN/MatMul_9_grad/tuple/control_dependency_1, gradients/RNN/MatMul_8_grad/tuple/control_dependency_1, gradients/RNN/MatMul_7_grad/tuple/control_dependency_1, gradients/RNN/MatMul_6_grad/tuple/control_dependency_1, gradients/RNN/MatMul_5_grad/tuple/control_dependency_1, gradients/RNN/MatMul_4_grad/tuple/control_dependency_1, gradients/RNN/MatMul_3_grad/tuple/control_dependency_1, gradients/RNN/MatMul_2_grad/tuple/control_dependency_1, gradients/RNN/MatMul_1_grad/tuple/control_dependency_1, gradients/RNN/MatMul_grad/tuple/control_dependency_1)]]
   [[Node: gradients/AddN_3/_2883 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_6304_gradients/AddN_3", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
Traceback (most recent call last):

"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_neg', True, '')


import melt
import conf 
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS

import vocabulary
  
class ShowAndTell(object):
  def __init__(self):
    super(ShowAndTell, self).__init__()
    self.sess = tf.InteractiveSession()
    
    vocab_size = vocabulary.get_vocab_size()

    hidden_size = 256
    emb_dim = 256

    init_width = 0.5 / emb_dim
    with tf.device('/cpu:0'):
      self.emb = tf.Variable(
        tf.random_uniform(
          [vocab_size, emb_dim], 
          -init_width, 
          init_width),
        name="emb")
      self.bemb = melt.init_bias([emb_dim], name='bemb')
    
    self.encode_img_W = tf.Variable(tf.random_uniform([IMAGE_FEATURE_LEN, hidden_size], -0.1, 0.1), name='encode_img_W')
    self.encode_img_b = melt.init_bias([hidden_size], name='encode_img_b')

    with tf.device('/cpu:0'):
      self.embed_word_W = tf.Variable(tf.random_uniform([emb_dim, vocab_size], -0.1, 0.1), name='embed_word_W')
      self.embed_word_b = melt.init_bias([vocab_size], name='embed_word_b')

    self.lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    self.n_lstm_steps = TEXT_MAX_WORDS + 2

    self.activation = tf.nn.relu
    
  def compute_seq_loss(self, image_emb, text):
    batch_size = tf.shape(text)[0]
    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text, pad])

    mask = tf.zeros_like(text)
    mask = tf.cast(tf.greater(text, mask), tf.float32)

    state = tf.zeros([batch_size, self.lstm.state_size])

    #loss = 0.0
    loss = tf.zeros([batch_size, 1])
    with tf.variable_scope("RNN"):
      for i in range(self.n_lstm_steps): 
        #print i
        if i == 0:
          current_emb = image_emb
        else:
          #with tf.device("/cpu:0"):
          current_emb = tf.nn.embedding_lookup(self.emb, text[:,i-1]) + self.bemb
        if i > 0 : tf.get_variable_scope().reuse_variables()
        #tf.get_variable_scope().reuse_variables()
        output, state = self.lstm(current_emb, state) # (batch_size, dim_hidden)

        if i > 0: 
          labels = text[:, i]
          with tf.device('/cpu:0'):
            logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, text[:, i])
          #[batch_size,]
          cross_entropy = cross_entropy * mask[:,i]
          #current_loss = tf.reduce_sum(cross_entropy)
          current_loss = tf.reshape(cross_entropy, [batch_size, 1])
          loss = loss + current_loss
      #loss = loss / tf.reduce_sum(mask[:,1:])
      #----@NOTICE must keep dims
      #[batch_size, 1]
      loss = loss / tf.reduce_sum(mask[:,1:], 1, keep_dims=True)
      #loss = loss / (TEXT_MAX_WORDS + 1)
    return loss
  
  # #it seems withot negative sample, will prefer common text
  #for ranking loss, margin 0.1 seems not work, margin 1 is ok, might try direct ranking loss without margin log loss? @TODO
  def build_graph(self, image_feature, text, neg_text):
    image_emb = tf.nn.xw_plus_b(image_feature, self.encode_img_W, self.encode_img_b)
    pos_loss = self.compute_seq_loss(image_emb, text)

    if FLAGS.use_neg:
      neg_losses = []
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        tf.get_variable_scope().reuse_variables()
        neg_text_i = neg_text[:,i,:]
        neg_loss = self.compute_seq_loss(image_emb, neg_text_i)
        neg_losses.append(neg_loss)

      neg_losses = tf.concat(1, neg_losses)

      #[batch_size, num_neg_text] @NOTICE should input should be pos_score and neg_score which is the larger the better
      loss = melt.hinge_loss(-pos_loss, -neg_losses, FLAGS.margin)
      scores = tf.concat(1, [pos_loss, neg_losses])
    else:
      loss = tf.reduce_mean(pos_loss)
      scores = tf.concat(1, [pos_loss, pos_loss])

    return loss, scores


  def build_train_graph(self, image_feature, text, neg_text):
    return self.build_graph(image_feature, text, neg_text)