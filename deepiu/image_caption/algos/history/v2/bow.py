#!/usr/bin/env python
# ==============================================================================
#          \file   bow.py
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
flags.DEFINE_integer('emb_dim', 1024, 'embedding dim for each word, default emb dim is 256 see if 1024 will be better')
flags.DEFINE_integer('hidden_size', 1024, 'hidden size')
flags.DEFINE_float('margin', 0.5, 'margin for rankloss')
flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")
flags.DEFINE_float('weight_mean', 0.0001, 'weight mean')
flags.DEFINE_float('weight_stddev', 0.5,  
                                  """weight stddev, 
                                     @Notice if use bias then small stddev like 0.01 might not lead to convergence, 
                                     causing layer weight value always be 0""")
flags.DEFINE_boolean('bias', False, 'Whether to use bias.')

flags.DEFINE_string('combiner', 'sum', '')
flags.DEFINE_boolean('exclude_zero_index', True, 'wether really exclude the first row(0 index)')

#dynamic_batch_length or fixed_batch_length
#combiner can be sum or mean
#always use mask (asume 0 emb index to be 0 vector)
#but now not implement this, will do exp to show if ok @TODO
#for fixed_batch_length sum and mean should be same so just use sum

import tensorflow.contrib.slim as slim
import melt
import melt.slim 

#need to parse FLAGS.vocab so must be in module global path ccan not be inside Model.__init__
import vocabulary

#@TODO move to algo and Make it  Cbow() , use  algo_factor to gen model 
class Bow(object):
  def __init__(self, is_training=True):
    super(Bow, self).__init__()

    self.is_training = is_training

    if not FLAGS.dynamic_batch_length and not FLAGS.exclude_zero_index and FLAGS.combiner != 'sum':
      raise ValueError("""dynamic_batch_length=False, exclude_zero_index=False,
                          must use sum combiner(predictor will assume this also), 
                          but input combiner is:""", FLAGS.combiner)

    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    #+1 for other derived class might use additional end id
    vocab_size = vocabulary.get_vocab_size() + 1
    #vocab_size = vocabulary.get_vocab_size()
    print('bow vocab_size:', vocab_size)
    self.vocab_size = vocab_size
    #if not cpu and on gpu run and using adagrad, will fail
    #also this will be more safer, since emb is large might exceed gpu mem
    with tf.device('/cpu:0'):
      self.emb = melt.variable.get_weights_uniform('emb', [vocab_size, emb_dim], -init_width, init_width)

    if is_training and FLAGS.debug:
      tf.histogram_summary('debug-emb_0', tf.gather(self.emb, 0))
      tf.histogram_summary('debug-emb_nv', tf.gather(self.emb, 7))
      ##it seems gpu will not fail if exceeds bound
      # tf.histogram_summary('debug-emb_1k', tf.gather(self.emb, 1000))
      # tf.histogram_summary('debug-emb_1w', tf.gather(self.emb, 10000))
      # tf.histogram_summary('debug-emb_10w', tf.gather(self.emb, 100000))
      tf.histogram_summary('debug-emb_middle', tf.gather(self.emb, vocab_size // 2))
      tf.histogram_summary('debug-emb_end', tf.gather(self.emb, vocab_size - 1))
      tf.histogram_summary('debug-emb_end2', tf.gather(self.emb, vocab_size - 2))

    self.activation = melt.activations[FLAGS.activation]

  def forward_image_layers(self, image_feature):
    dims = [FLAGS.hidden_size, FLAGS.hidden_size]
    #return melt.slim.mlp(image_feature, dims, self.activation, scope='image_mlp')
    return melt.layers.mlp_nobias(image_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='image_mlp')

  def forward_text_layers(self, text_feature):
    dims = [FLAGS.hidden_size, FLAGS.hidden_size]
    #return melt.slim.mlp(text_feature, dims, self.activation, scope='text_mlp')
    return melt.layers.mlp_nobias(text_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='text_mlp')

  def forward_text_feature(self, text_feature):
    text_feature = self.forward_text_layers(text_feature)
    text_feature = tf.nn.l2_normalize(text_feature, 1)
    return text_feature	

  def embedding_lookup(self, index):
    #with tf.device('/cpu:0'):
    return melt.batch_masked_embedding_lookup(self.emb, 
                                              index, 
                                              combiner=FLAGS.combiner, 
                                              exclude_zero_index=FLAGS.exclude_zero_index)

  def gen_text_feature(self, text):
    """
    this common interface, ie may be lstm will use other way to gnerate text feature
    """
    return self.embedding_lookup(text)

  def forward_text(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.embedding_lookup(text)
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

  def build_graph(self, image_feature, text, neg_text, lookup_negs_once=True):
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
      text_feature = self.gen_text_feature(text)
      pos_score = self.compute_image_text_sim(normed_image_feature, text_feature)
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]
      tf.get_variable_scope().reuse_variables()
      if lookup_negs_once:
        neg_text_feature = self.gen_text_feature(neg_text)
      neg_scores_list = []
      
      num_negs = neg_text.get_shape()[1]
      for i in xrange(num_negs):
        if lookup_negs_once:
          neg_text_feature_i = neg_text_feature[:, i, :]
        else:
          neg_text_feature_i = self.gen_text_feature(neg_text[:, i, :])
        neg_scores_i = self.compute_image_text_sim(normed_image_feature, neg_text_feature_i)
        neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(1, neg_scores_list)

      #---------------rank loss
      #[batch_size, 1 + num_negs]
      scores = tf.concat(1, [pos_score, neg_scores])
      loss = melt.hinge_loss(pos_score, neg_scores, FLAGS.margin)
    
      self.scores = scores
    return loss

  def build_train_graph(self, image_feature, text, neg_text, lookup_negs_once=True):
    return self.build_graph(image_feature, text, neg_text, lookup_negs_once)
