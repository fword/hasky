#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   discriminant_trainer.py
#        \author   chenghuige  
#          \date   2016-09-22 22:39:10.084671
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_size', 1024, 'hidden size')
flags.DEFINE_float('margin', 0.5, 'margin for rankloss when rank_loss is hinge loss')
flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")

flags.DEFINE_boolean('bias', False, 'wether to use bias. Not using bias can speedup a bit')
flags.DEFINE_string('rank_loss', 'hinge', 'use hinge(hinge_loss) or cross(cross_entropy_loss) or hinge_cross(subtract then cross)')

  
import tensorflow.contrib.slim as slim
import melt
import melt.slim 

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding

class DiscriminantTrainer(object):
  """
  Only need to set self.gen_text_feature
  """
  def __init__(self, is_training=True, is_predict=False):
    super(DiscriminantTrainer, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict
    self.gen_text_feature = None

    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    self.vocab_size = vocab_size
    #if not cpu and on gpu run and using adagrad, will fail  TODO check why
    #also this will be more safer, since emb is large might exceed gpu mem   
    #with tf.device('/cpu:0'):
    #  #NOTICE if using bidirectional rnn then actually emb_dim is emb_dim / 2, because will at last step depth-concatate output fw and bw vectors
    #  self.emb = melt.variable.get_weights_uniform('emb', [vocab_size, emb_dim], -init_width, init_width)
    self.emb = embedding.get_embedding_cpu('emb')

    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self.activation = melt.activations[FLAGS.activation]

    self.weights_initializer = tf.random_uniform_initializer(-FLAGS.initializer_scale, FLAGS.initializer_scale)
    self.biases_initialzier = melt.slim.init_ops.zeros_initializer if FLAGS.bias else None

  def forward_image_layers(self, image_feature):
    dims = [FLAGS.hidden_size, FLAGS.hidden_size]
    return melt.slim.mlp(image_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier, 
                         scope='image_mlp')
    #return melt.layers.mlp_nobias(image_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='image_mlp')

  def forward_text_layers(self, text_feature):
    dims = [FLAGS.hidden_size, FLAGS.hidden_size]
    return melt.slim.mlp(text_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier, 
                         scope='text_mlp')
    #return melt.layers.mlp_nobias(text_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='text_mlp')

  def forward_text_feature(self, text_feature):
    text_feature = self.forward_text_layers(text_feature)
    text_feature = tf.nn.l2_normalize(text_feature, 1)
    return text_feature	

  def forward_text(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.gen_text_feature(text, self.emb)
    text_feature = self.forward_text_feature(text_feature)
    return text_feature

  def forward_image_feature(self, image_feature):
    """
    Args:
      image: batch image [batch_size, image_feature_len]
    """
    image_feature = self.forward_image_layers(image_feature)
    image_feature = tf.nn.l2_normalize(image_feature, 1)
    return image_feature

  def compute_image_text_sim(self, normed_image_feature, text_feature):
    #[batch_size, hidden_size]
    normed_text_feature = self.forward_text_feature(text_feature)
    #[batch_size,1] <= [batch_size, hidden_size],[batch_size, hidden_size]
    return  melt.element_wise_cosine(normed_image_feature, normed_text_feature, nonorm=True)

  def build_graph(self, image_feature, text, neg_text, lookup_negs_once=False):
    """
    Args:
    image_feature: [batch_size, IMAGE_FEATURE_LEN]
    text: [batch_size, MAX_TEXT_LEN]
    neg_text: [batch_size, num_negs, MAXT_TEXT_LEN]
    """
    with tf.variable_scope("image_text_sim"):
      #-------------get image feature
      #[batch_size, hidden_size] <= [batch_size, IMAGE_FEATURE_LEN] 
      normed_image_feature = self.forward_image_feature(image_feature)

      #--------------get image text sim as pos score
      #[batch_size, emb_dim] -> [batch_size, text_MAX_WORDS, emb_dim] -> [batch_size, emb_dim]
      text_feature = self.gen_text_feature(text, self.emb)
      tf.add_to_collection('text_feature', text_feature)

      pos_score = self.compute_image_text_sim(normed_image_feature, text_feature)
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]
      tf.get_variable_scope().reuse_variables()
      if lookup_negs_once:
        neg_text_feature = self.gen_text_feature(neg_text, self.emb)
      neg_scores_list = []
      
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        if lookup_negs_once:
          neg_text_feature_i = neg_text_feature[:, i, :]
        else:
          neg_text_feature_i = self.gen_text_feature(neg_text[:, i, :], self.emb)
        neg_scores_i = self.compute_image_text_sim(normed_image_feature, neg_text_feature_i)
        neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(neg_scores_list, 1)

      #---------------rank loss
      #[batch_size, 1 + num_negs]
      scores = tf.concat([pos_score, neg_scores], 1)
      #may be turn to prob is and show is 
      #probs = tf.sigmoid(scores)

      if FLAGS.rank_loss == 'hinge':
        loss = melt.hinge_loss(pos_score, neg_scores, FLAGS.margin)
      elif FLAGS.rank_loss == 'cross':
        loss = melt.cross_entropy_loss(scores, num_negs)
      else: 
        loss = melt.hinge_cross_loss(pos_score, neg_scores)
      
      tf.add_to_collection('scores', scores)
    return loss

  
