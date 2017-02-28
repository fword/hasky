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

flags.DEFINE_boolean('use_neg', True, '')
flags.DEFINE_integer('num_samples', 512, 'num samples of neg word from vocab')


import melt
import conf 
from conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS

import vocabulary

cell = tf.nn.rnn_cell.BasicLSTMCell
#cell = tf.nn.rnn_cell.GRUCell
  
class ShowAndTell(object):
  def __init__(self):
    super(ShowAndTell, self).__init__()
    self.sess = tf.InteractiveSession()
    
    vocab_size = vocabulary.get_vocab_size()

    self.emb_dim = emb_dim = hidden_size = 256

    #init_width = 0.5 / emb_dim
    init_width = 0.1
    with tf.device('/cpu:0'):
      self.emb = tf.Variable(tf.random_uniform([vocab_size, emb_dim], 
          -init_width, 
          init_width),
        name="emb")
      self.bemb = melt.init_bias([emb_dim], name='bemb')
    
    self.encode_img_W = tf.Variable(tf.random_uniform([IMAGE_FEATURE_LEN, hidden_size], -0.1, 0.1), name='encode_img_W')
    self.encode_img_b = melt.init_bias([hidden_size], name='encode_img_b')

    with tf.device('/cpu:0'):
      self.embed_word_W = tf.Variable(tf.random_uniform([emb_dim, vocab_size], -0.1, 0.1), name='embed_word_W')
      self.embed_word_b = melt.init_bias([vocab_size], name='embed_word_b')

    self.cell = cell(hidden_size, state_is_tuple=True)
    #------GRUCell has no arg state_is_tuple
    #self.cell = cell(hidden_size)

    self.activation = tf.nn.relu

    num_samples = FLAGS.num_samples

    #@TODO move to melt  def prepare_sampled_softmax_loss(num_samples, vocab_size, hidden_size)
    #return output_projection, softmax_loss_function
    #also consider candidate sampler 
    self.softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < vocab_size:
      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(tf.transpose(self.embed_word_W), self.embed_word_b, inputs, labels, num_samples,
                                            vocab_size)

      self.softmax_loss_function = sampled_loss

  
  def _compute_seq_loss_baseline(self, image_emb, text):
    """
    This is base line
    """
    batch_size = tf.shape(text)[0]
    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text, pad]) 
    num_steps = TEXT_MAX_WORDS + 2

    mask = tf.cast(tf.sign(text), dtype=tf.float32)

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = tf.nn.embedding_lookup(self.emb, text[:,i - 1]) + self.bemb
        if i > 0 : tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(current_emb, state) # (batch_size, dim_hidden)

        if i > 0: 
          labels = text[:, i]
          with tf.device('/cpu:0'):
            logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, labels)
          #[batch_size,]
          cross_entropy = cross_entropy * mask[:,i]
          current_loss = tf.reshape(cross_entropy, [batch_size, 1])
          loss = loss + current_loss
      #[batch_size, 1]
      loss = loss / tf.reduce_sum(mask[:,1:], 1, keep_dims=True)
    return loss

  def _compute_seq_loss2(self, image_emb, text):
    """
    diff with _compute_se_loss_baseline is  
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    pre fetch all embedding for sequence
    """
    batch_size = tf.shape(text)[0]
    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text, pad]) 
    num_steps = TEXT_MAX_WORDS + 2

    mask = tf.cast(tf.sign(text), dtype=tf.float32)

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = inputs[:, i - 1, :]
        if i > 0 : tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(current_emb, state) # (batch_size, dim_hidden)

        if i > 0: 
          labels = text[:, i]
          with tf.device('/cpu:0'):
            logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
          cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, labels)
          #[batch_size,]
          cross_entropy = cross_entropy * mask[:,i]
          current_loss = tf.reshape(cross_entropy, [batch_size, 1])
          loss = loss + current_loss
      #[batch_size, 1]
      loss = loss / tf.reduce_sum(mask[:,1:], 1, keep_dims=True)
    return loss

  def _compute_seq_loss3(self, image_emb, text):
    """
    diff with _compute_se_loss_baseline is  
    reove all pad
    # 的/，/好/，/好/，/你/的/，/好
    """
    batch_size = tf.shape(text)[0]
    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    num_steps = TEXT_MAX_WORDS

    mask = tf.cast(tf.sign(text), dtype=tf.float32)

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = inputs[:, i - 1, :]
        if i > 0 : tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(current_emb, state) # (batch_size, dim_hidden)

        labels = text[:, i]
        with tf.device('/cpu:0'):
          logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, labels)
        #[batch_size,]
        cross_entropy = cross_entropy * mask[:,i]
        current_loss = tf.reshape(cross_entropy, [batch_size, 1])
        loss = loss + current_loss
      #[batch_size, 1]
      loss = loss / tf.reduce_sum(mask, 1, keep_dims=True)
    return loss


  def _compute_seq_loss4(self, image_emb, text):
  #def compute_seq_loss(self, image_emb, text):
    """
    diff with _compute_se_loss_baseline is  
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    pre fetch all embedding for sequence

    compare the diff with _compute_seq_loss3
    seems [pad, text] ,here pad before text is needed 
    will generate good sentences but not seems very realted to image like no  chi..
    """
    batch_size = tf.shape(text)[0]
    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 
    num_steps = TEXT_MAX_WORDS + 1

    mask = tf.cast(tf.sign(text), dtype=tf.float32)

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = inputs[:, i - 1, :]
        if i > 0 : tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(current_emb, state) # (batch_size, dim_hidden)

        #if i > 0:
        labels = text[:, i]
        with tf.device('/cpu:0'):
          logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_words, labels)
        #[batch_size,]
        cross_entropy = cross_entropy * mask[:,i]
        current_loss = tf.reshape(cross_entropy, [batch_size, 1])
        loss = loss + current_loss
      #[batch_size, 1]
      loss = loss / tf.reduce_sum(mask[:,1:], 1, keep_dims=True)
    return loss

  def _compute_seq_loss5(self, image_emb, text):
  #def compute_seq_loss(self, image_emb, text):
    """
    this one seems perform well
    but notice output loss will be [batch_size * num_steps, 1]
    also will need about 0.8+ epoch to have resonable generated words like chi
    <p> generated:[ 好/的/啊/，/我/也/想/吃/了/， ] </p>
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    outputs = []
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = inputs[:, i - 1, :]
          tf.get_variable_scope().reuse_variables()

        cell_output, state = self.cell(current_emb, state) 
        outputs.append(cell_output)

      #[batch_size * num_steps, emb_dim]
      output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
      logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
      targets = text
      mask = tf.cast(tf.sign(text), dtype=tf.float32)
      loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.reshape(mask, [batch_size * num_steps])])
      
      #[batch_size * num_steps, 1]
      loss = tf.reshape(loss, [-1, 1])
    return loss

  def _compute_seq_loss6(self, image_emb, text):
    """
    use sequence loss 
    and also show with out first pad will prefer 
    generated:[ 的/的/，/，/，/，/，/，/，/， ]
    it is because in show_and_tell_predictor.py assume to have pad at first when decode
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS 

    state = self.cell.zero_state(batch_size, tf.float32)

    loss = tf.zeros([batch_size, 1])
    inputs = tf.nn.embedding_lookup(self.emb, text) + self.bemb
    outputs = []
    with tf.variable_scope("RNN"):
      for i in range(num_steps): 
        if i == 0:
          current_emb = image_emb
        else:
          current_emb = inputs[:, i - 1, :]
          tf.get_variable_scope().reuse_variables()

        cell_output, state = self.cell(current_emb, state) 
        outputs.append(cell_output)
      
      #[batch_size * num_steps, emb_dim]
      output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
      logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
      targets = text
      mask = tf.cast(tf.sign(text), dtype=tf.float32)
      loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
        [tf.reshape(targets, [-1])],
        [tf.reshape(mask, [batch_size * num_steps])])
      loss = tf.reshape(loss, [-1, 1])
    return loss

  def _compute_seq_loss7(self, image_emb, text):
  #def compute_seq_loss(self, image_emb, text):
    """
    use tf.nn.rnn
    <p> generated:[ 好/吃/的/样子/，/我/想/吃/了/， ] </p>
    faster 0.027 then now standard
    compute_seq_loss
    notice then can use
    loss = melt.reduce_mean(pos_loss)
    so do not consider zeros
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state)
    
    output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
    logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
    targets = text
    
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    #will prefer predicting 0, for will consider pad 0 as loss
    #mask = tf.ones_like(text, dtype=tf.float32)
    
    loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
      [tf.reshape(targets, [-1])],
      [tf.reshape(mask, [batch_size * num_steps])])
    loss = tf.reshape(loss, [-1, 1])
    
    return loss

  def _compute_seq_loss8(self, image_emb, text):
    """
    same as _compute_seq_loss7, use tf.nn.rnn
    but will modify input of sequence_loss_by_example
    to make it output loss with shape [batch_size, 1] 
    verified shape ok but not good result 
    <p> generated:[ 666/666/666/666/666/666/666/666/你/， ] </p>
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state)
    
    output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
    logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
    logits = [logit_ for logit_ in tf.split(0, num_steps, logits)]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights)
    loss = tf.reshape(loss, [-1, 1])
    
    #self.debug_result = loss

    return loss

  def _compute_seq_loss9(self, image_emb, text):
    """
    same as _compute_seq_loss8
    but use average_across_timesteps=True
    <p> generated:[ 你好/666/666/好/好/666/666/666/666/， ] </p>
    still not good
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state)
    
    #[batch_size * num_steps, emb_dim]
    output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
    logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
    logits = [logit_ for logit_ in tf.split(0, num_steps, logits)]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True)
    loss = tf.reshape(loss, [-1, 1])

    return loss
  
  def _compute_seq_loss10(self, image_emb, text):
    """
    seems logits list calc wrong in _compute_seq_loss9 and _compute_seq_loss8
    fix this problem 
    <p> generated:[ 好/萌/啊/！/！/！/！/！/！/！ ] </p>
    sentence language model ok
    but seems not very related to input image, no chi output
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state)
    
    #[batch_size * num_steps, emb_dim]
    #output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
    #logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
    #logits = [logit_ for logit_ in tf.split(0, num_steps, logits)]
    #ref to mine/ipynotebook/tensorflow/rnn/dynamic_rnn.ipynb
    #above is wrong another way is
    #logits = tf.reshape(logits, [batch_size, num_steps])
    #logits = [logit_ for logit_ in tf.split(1, num_steps, logits)]
    logits = [tf.matmul(output_, self.embed_word_W) + self.embed_word_b 
              for output_ in outputs]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    #all ones will make generate []..., for it will attention on predicting 0
    #mask = tf.ones_like(text, dtype=tf.float32)
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    #loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights,
    #average_across_timesteps=True)
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights)
    loss = tf.reshape(loss, [-1, 1])

    return loss

  def _compute_seq_loss11(self, image_emb, text):
    """
    <p> generated:[ 好/大/的/花/，/好/大/的/，/， ] </p>
    <p> generated:[ 666/，/我/也/想/吃/了/。/。/。 ] </p>
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    pad = tf.zeros([batch_size, 1], dtype=text.dtype)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state)
    
    logits = [tf.matmul(output_, self.embed_word_W) + self.embed_word_b 
              for output_ in outputs]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    #all 3 lists(logits, targets, weights) length is num_steps
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True)
    loss = tf.reshape(loss, [-1, 1])

    return loss

  def _compute_seq_loss12(self, image_emb, text):
    """
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    pad = tf.zeros([batch_size, 1], dtype=text.dtype)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    inputs = [tf.squeeze(input_, [1])
               for input_ in tf.split(1, num_steps, inputs)]
    
    mask = tf.cast(tf.sign(text), tf.float32)
    sequence_length = tf.reduce_sum(mask, 1) + 1
    outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=state, sequence_length=sequence_length)
    
    #[batch_size * num_steps, emb_dim]
    output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_dim])
    logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b
    logits = tf.reshape(logits, [batch_size, num_steps, -1])
    logits = [tf.squeeze(logit_, [1]) 
              for logit_ in tf.split(1, num_steps, logits)]

    #logits = [tf.matmul(output_, self.embed_word_W) + self.embed_word_b
    #          for output_ in outputs]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    #all 3 lists(logits, targets, weights) length is num_steps
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True)
    loss = tf.reshape(loss, [-1, 1])

    return loss

  def _compute_seq_loss13(self, image_emb, text):
  #def compute_seq_loss(self, image_emb, text):
    """
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    pad = tf.zeros([batch_size, 1], dtype=text.dtype)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    #list of [batch_size, emb_dim]
    #inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    
    mask = tf.cast(tf.sign(text), dtype=tf.float32)
    sequence_length = tf.reduce_sum(mask, 1) + 1
    outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=state, sequence_length=sequence_length)
    
    #[batch_size * num_steps, emb_dim]
    #logits = [tf.matmul(output_, self.embed_word_W) + self.embed_word_b
    #          for output_ in outputs]
    output = tf.reshape(outputs, [-1, self.emb_dim])
    with tf.device('/cpu:0'):
      logits = tf.matmul(output, self.embed_word_W) + self.embed_word_b if self.softmax_loss_function is None else output
    logits = tf.reshape(logits, [batch_size, num_steps, -1])
    logits = [tf.squeeze(logit_, [1]) 
              for logit_ in tf.split(1, num_steps, logits)]
    
    targets = [tf.squeeze(text_, [1]) 
               for text_ in tf.split(1, num_steps, text)]
    
    
    weights = [tf.squeeze(weight_, [1])
               for weight_ in tf.split(1, num_steps, mask)]
    
    #all 3 lists(logits, targets, weights) length is num_steps
    loss = tf.nn.seq2seq.sequence_loss_by_example(logits, targets, weights,
                                                  average_across_timesteps=True,
                                                  softmax_loss_function=self.softmax_loss_function)
    loss = tf.reshape(loss, [-1, 1])

    return loss

  def compute_seq_loss(self, image_emb, text):
    """
    same ass 7
    but use dynamic rnn
    """
    batch_size = tf.shape(text)[0]
    num_steps = TEXT_MAX_WORDS + 1

    #@TODO get dtype from text so text can be tf.int32
    pad = tf.zeros([batch_size, 1], dtype=tf.int64)
    text = tf.concat(1, [pad, text]) 

    #@TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32)

    self.initial_state = state

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(self.emb, text[:,:num_steps - 1]) + self.bemb
    #[batch_size, num_steps, emb_dim] image_emp( [batch_size, emb_dim] ->
    #[batch_size, 1, emb_dim]) before concat
    inputs = tf.concat(1, [tf.expand_dims(image_emb, 1), inputs])
    
    mask = tf.sign(text)
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
      [tf.reshape(mask, [batch_size * num_steps])],
      softmax_loss_function=self.softmax_loss_function)
    
    loss = tf.reshape(loss, [-1, 1])
    #--------@TODO seems using below and tf.reduce_mean will generate not as good as above loss and melt.reduce_mean 
    #--if no bug the diff shold be per example(per step) loss and per single step loss
    #loss = melt.reduce_mean_with_mask(tf.reshape(loss, [batch_size, num_steps]), mask, 
    #                                  reduction_indices=1, keep_dims=True)
    
    return loss

  # #it seems withot negative sample, will prefer common text
  #for ranking loss, margin 0.1 seems not work, margin 1 is ok, might try
  #direct ranking loss without margin log loss?  @TODO
  #@TODO only return loss, remove socres, you can debug using self.scores =
  #scores and outside code builder.scores
  def build_graph(self, image_feature, text, neg_text=None):
    image_emb = tf.nn.xw_plus_b(image_feature, self.encode_img_W, self.encode_img_b)
    pos_loss = self.compute_seq_loss(image_emb, text)

    if neg_text is not None and FLAGS.use_neg:
      neg_losses = []
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        tf.get_variable_scope().reuse_variables()
        neg_text_i = neg_text[:,i,:]
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
      loss = melt.reduce_mean(pos_loss)
      scores = tf.concat(1, [pos_loss, pos_loss])

    self.scores = scores
    return loss

  def build_train_graph(self, image_feature, text, neg_text):
    return self.build_graph(image_feature, text, neg_text)