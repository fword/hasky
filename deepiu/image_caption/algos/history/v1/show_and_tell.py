#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
lstm based generative model

@TODO try to use seq2seq.py 
* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_neg', False, '')
flags.DEFINE_boolean('per_example_loss', False, """if False just per step loss, 
                                                   True per example loss(per example per step)
                                                   if use_neg=True and dynamic_batch_length=True 
                                                   then must set per_example_loss=True""")
flags.DEFINE_integer('num_sampled', 512, 'num samples of neg word from vocab')
flags.DEFINE_float('keep_prob', 0.5, '')

flags.DEFINE_integer('num_layers', 2, '')

import melt
import conf 
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS

import vocabulary


cell = tf.nn.rnn_cell.BasicLSTMCell
#cell = tf.nn.rnn_cell.GRUCell
  
class ShowAndTell(object):
  """
  ShowAndTell class is a trainer class
  but has is_training mark for ShowAndTell predictor will share some code here
  """
  def __init__(self, is_training=True):
    super(ShowAndTell, self).__init__()

    if is_training:
      print('num_sampled:', FLAGS.num_sampled)
      print('use_neg:', FLAGS.use_neg)
      print('per_example_loss:', FLAGS.per_example_loss)

    vocab_size = vocabulary.get_vocab_size()
    self.vocab_counts_list = [vocabulary.vocab.freq(i) for i in xrange(vocab_size)]
    self.vocab_counts_list.append(1)

    #vocabe_size + 1 add one for store end id
    vocab_size += 1
    self.vocab_size = vocab_size
    self.end_id = vocab_size - 1

   
    #self.emb_dim = emb_dim = hidden_size = 256
    self.emb_dim = emb_dim = hidden_size = 1024

    init_width = 0.5 / emb_dim
    #init_width = 0.1
    with tf.device('/cpu:0'):
      self.emb = melt.variable.get_weights_uniform(
        'emb', [vocab_size, emb_dim], -init_width, init_width)
      self.bemb = melt.variable.get_bias('bemb', [emb_dim])
      
      self.embed_word_W = melt.variable.get_weights_uniform('embed_word_W', [emb_dim, vocab_size], -0.1, 0.1)
      self.embed_word_b = melt.variable.get_bias('embed_word_b', [vocab_size])
    
    self.encode_img_W = melt.variable.get_weights_uniform('encode_img_W', [IMAGE_FEATURE_LEN, hidden_size], -0.1, 0.1)
    self.encode_img_b = melt.variable.get_bias('encode_img_b', [hidden_size])

    self.cell = cell(hidden_size, state_is_tuple=True)
    if is_training and FLAGS.keep_prob < 1:
     self.cell = tf.nn.rnn_cell.DropoutWrapper(
         self.cell, output_keep_prob=FLAGS.keep_prob)
    self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * FLAGS.num_layers, state_is_tuple=True)
    #------GRUCell has no arg state_is_tuple
    #self.cell = cell(hidden_size)

    self.activation = tf.nn.relu

    num_sampled = FLAGS.num_sampled

    #@TODO move to melt  def prepare_sampled_softmax_loss(num_sampled, vocab_size, hidden_size)
    #return output_projection, softmax_loss_function
    #also consider candidate sampler 
    self.softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_sampled > 0 and num_sampled < vocab_size:
      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])

          sampled_values = tf.nn.fixed_unigram_candidate_sampler(
              true_classes=labels,
              num_true=1,
              num_sampled=num_sampled,
              unique=True,
              range_max=vocab_size,
              distortion=0.75,
              unigrams=self.vocab_counts_list)
              
          return tf.nn.sampled_softmax_loss(tf.transpose(self.embed_word_W), 
                                            self.embed_word_b, 
                                            inputs, 
                                            labels, 
                                            num_sampled, 
                                            vocab_size,
                                            sampled_values=sampled_values)

      self.softmax_loss_function = sampled_loss

  def compute_seq_loss(self, image_emb, text, is_training=True):
    """
    same ass 7
    but use dynamic rnn
    """
    #notice here must use tf.shape not text.get_shape()[0], because it is dynamic shape, known at runtime
    batch_size = tf.shape(text)[0]
    
    zero_pad = tf.zeros([batch_size, 1], dtype=text.dtype)
    
    #add zero before sentence to avoid always generate A... 
    #add zero after sentence to make sure end mark will not exceed boundary incase your input sentence is long with out 0 padding at last
    #text = tf.concat(1, [zero_pad, text, zero_pad])
    text = tf.concat(1, [zero_pad, text, zero_pad])

    sequence_length = melt.length(text) + 1
    text = melt.dynamic_append_with_length(text, sequence_length, tf.constant(self.end_id, dtype=text.dtype)) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    self.initial_state = state

    #print('melt.last_dimension(text)', melt.last_dimension(text))

    #[batch_size, num_steps - 1, emb_dim], remove last col
    #notice tf 10.0 now do not support text[:,:-1] @TODO may change to that if tf support in future
    #now the hack is to use last_dimension wich will inside use static shape notice dynamic shape like tf.shape not work!
    #using last_dimension is static type! Konwn on graph construction not dynamic runtime
    #inputs = tf.nn.embedding_lookup(self.emb, text[:,:melt.last_dimension(text) - 1]) + self.bemb
    # TypeError("Using a `tf.Tensor` as a Python `bool` is not allowed. "
    #inputs = tf.nn.embedding_lookup(self.emb, text[:,:tf.shape(text)[1] - 1]) + self.bemb
    #can see ipynotebook/dynamic_length.npy
    #well this work..
    #num_steps = tf.shape(text)[1]
    inputs = tf.nn.embedding_lookup(self.emb, melt.exclude_last_col(text)) + self.bemb
    
    if is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
    
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])


    
    mask = tf.sign(text)
    # +1 for zero pad before sentence!
    sequence_length = tf.reduce_sum(mask, 1) + 1

    outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=state, 
                                       sequence_length=sequence_length)
    self.final_state = state
    
    #@TODO now looks like this version is much faster then using like _compute_seq_loss13
    #but still there are much un necessary calculations like mat mul for all batch_size * num steps ..
    #can we speed up by not calc loss for mask[pos] == 0 ?
    output = tf.reshape(outputs, [-1, self.emb_dim])
    with tf.device('/cpu:0'):
      logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b if self.softmax_loss_function is None else output
    targets = text
    
    mask = tf.cast(mask, dtype=tf.float32)
    
    loss = tf.nn.seq2seq.sequence_loss_by_example(
      [logits],
      [tf.reshape(targets, [-1])],
      [tf.reshape(mask, [-1])],
      softmax_loss_function=self.softmax_loss_function)
    
    #--------@TODO seems using below and tf.reduce_mean will generate not as good as above loss and melt.reduce_mean 
    #--if no bug the diff shold be per example(per step) loss and per single step loss
    if (not is_training) or FLAGS.per_example_loss:
      loss = melt.reduce_mean_with_mask(tf.reshape(loss, [batch_size, -1]), mask, 
                                        reduction_indices=1, keep_dims=True)
    else:
      #if use this the will be [batch_size * num_steps, 1], so for use negs, could not use dynamic length mode
      loss = tf.reshape(loss, [-1, 1])
    
    return loss

  # #it seems withot negative sample, will prefer common text
  #for ranking loss, margin 0.1 seems not work, margin 1 is ok, might try
  #direct ranking loss without margin log loss?  @TODO
  #@TODO only return loss, remove socres, you can debug using self.scores =
  #scores and outside code builder.scores
  def build_graph(self, image_feature, text, neg_text=None, is_training=True):
    image_emb = tf.nn.xw_plus_b(image_feature, self.encode_img_W, self.encode_img_b)
    pos_loss = self.compute_seq_loss(image_emb, text, is_training)

    if neg_text is not None and FLAGS.use_neg:
      neg_losses = []
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        tf.get_variable_scope().reuse_variables()
        neg_text_i = neg_text[:, i, :]
        neg_loss = self.compute_seq_loss(image_emb, neg_text_i)
        neg_losses.append(neg_loss)

      neg_losses = tf.concat(1, neg_losses)

      #[batch_size, num_neg_text] @NOTICE input should be pos_score and
      #neg_score which is the larger the better
      loss = melt.hinge_loss(-pos_loss, -neg_losses, FLAGS.margin)
      scores = tf.concat(1, [pos_loss, neg_losses])
    else:
      #loss = tf.reduce_mean(pos_loss) 
      #use melt.reduce_mean if input is not [batch_size, 1] but [batch_size * num_steps, 1]
      if is_training:
        #[] per step loss
        if not FLAGS.per_example_loss:
          loss = melt.reduce_mean(pos_loss)
        else:
          loss = tf.reduce_mean(pos_loss)
      else:
        #[batch_size,] per example loss, if batch_size = 1 then []
        loss = tf.squeeze(pos_loss)

      scores = tf.concat(1, [pos_loss, pos_loss])

    self.scores = scores
    return loss

  def build_train_graph(self, image_feature, text, neg_text):
    return self.build_graph(image_feature, text, neg_text)