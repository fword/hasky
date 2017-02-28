#!/usr/bin/env python
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2016-08-18 00:56:57.071798
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('emb_dim', 256, 'embedding dim for each word')
flags.DEFINE_integer('hidden_size', 1024, 'hidden size')
flags.DEFINE_float('margin', 0.5, 'margin for rankloss')
flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")
flags.DEFINE_float('weight_mean', 0.0001, 'weight mean')
flags.DEFINE_float('weight_stddev', 0.5, 
    """weight stddev, 
    notice if use bias then small stddev like 0.01 might not lead to convergence, 
    causing layer weight value always be 0""")
flags.DEFINE_boolean('bias', False,
                         """Whether to use bias.""")


import tensorflow.contrib.slim as slim
import melt
from melt.variable import *
import melt.slim 

#need to parse FLAGS.vocab so must be in module global path ccan not be inside Model.__init__
import vocabulary

#@TODO move to algo and Make it  Cbow() , use  algo_factor to gen model 
class Model(object):
  def __init__(self):
    super(Model, self).__init__()
    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocab_size = vocabulary.get_vocab_size()
    self.vocab_size = vocab_size
    #if not cpu and on gpu run and using adagrad, will fail
    #also this will be more safer, since emb is large might exceed gpu mem
    with tf.device('/cpu:0'):
      self.emb = init_weights_uniform([vocab_size, emb_dim], -init_width, init_width, name='emb')
    self.activation = melt.activations[FLAGS.activation]

  def forward_image_layers(self, image_feature):
    dims = [1024, 1024]
    #return melt.slim.mlp(image_feature, dims, self.activation, scope='image_mlp')
    return melt.layers.mlp_nobias(image_feature, 1024, 1024, self.activation, scope='image_mlp')

  def forward_text_layers(self, text_feature):
    dims = [1024, 1024]
    #return melt.slim.mlp(text_feature, dims, self.activation, scope='text_mlp')
    return melt.layers.mlp_nobias(text_feature, 1024, 1024, self.activation, scope='text_mlp')

  def forward_text_feature(self, text_feature):
    text_feature = self.forward_text_layers(text_feature)
    text_feature = tf.nn.l2_normalize(text_feature, 1)
    return text_feature	
  
  def forward_text(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = melt.embedding_lookup_mean(self.emb, text, 1)
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

  def build_graph(self, image_feature, text, neg_text):
    """
    Args:
    image_feature: [batch_size, IMAGE_FEATURE_LEN]
    text: [batch_size, MAX_TEXT_LEN]
    neg_text: [batch_size, num_negs, MAXT_TEXT_LEN]
    """
    emb = self.emb
 
    with tf.variable_scope("image_text_sim"):
      #-------------get image feature
      #[batch_size, hidden_size] <= [batch_size, IMAGE_FEATURE_LEN] 
      normed_image_feature = self.forward_image_feature(image_feature)

      #--------------get image text sim as pos score
      #[batch_size, text_MAX_WORDS, emb_dim] -> [batch_size, emb_dim]
      normed_text_feature = melt.embedding_lookup_mean(self.emb, text, 1)
      pos_score = self.compute_image_text_sim(normed_image_feature, normed_text_feature)
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]
      tf.get_variable_scope().reuse_variables()
      neg_text_feature = melt.embedding_lookup_mean(emb, neg_text, 2)
      neg_scores_list = []
      
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        neg_text_feature_i = neg_text_feature[:,i,:]
        neg_scores_i = self.compute_image_text_sim(normed_image_feature, neg_text_feature_i)
        neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(1, neg_scores_list)

      #---------------rank loss
      #[batch_size, 1 + num_negs]
      scores = tf.concat(1, [pos_score, neg_scores])
      loss = melt.hinge_loss(pos_score, neg_scores, FLAGS.margin)
    
    return loss, scores
