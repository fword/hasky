#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_decoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:05.991481
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_sampled', 10000, 'num samples of neg word from vocab')
flags.DEFINE_boolean('log_uniform_sample', True, '')

flags.DEFINE_boolean('add_text_start', False, """if True will add <s> or 0 or GO before text 
                                              as first input before image, by default will be GO, make add_text_start==True if you use seq2seq""")
flags.DEFINE_boolean('zero_as_text_start', False, """if add_text_start, 
                                                    here True will add 0 False will add <s>
                                                    0 means the loss from image to 0/pad not considered""")
flags.DEFINE_boolean('go_as_text_start', True, """if add_text_start, 
                                                    here True will add 0 False will add <s>
                                                    0 means the loss from image to 0/pad not considered""")

flags.DEFINE_boolean('input_with_start_mark', False, """if input has already with <S> start mark""")
flags.DEFINE_boolean('input_with_end_mark', False, """if input has already with </S> end mark""")

flags.DEFINE_boolean('predict_with_end_mark', True, """if predict with </S> end mark""")

flags.DEFINE_float('length_normalization_factor', 1.0, """If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.  see tensorflow/models/im2text""")

flags.DEFINE_boolean('predict_use_prob', True, 'if True then use exp(logprob) and False will direclty output logprob')
flags.DEFINE_boolean('predict_no_sample', False, 'if True will use exact loss')
flags.DEFINE_integer('predict_sample_seed', 0, '')

import melt 
from deepiu.util import vocabulary
from deepiu.seq2seq.decoder import Decoder

class SeqDecodeMethod:
  max_prob = 0
  sample = 1
  full_sample = 2
  beam_search = 3

class RnnDecoder(Decoder):
  def __init__(self, is_training=True, is_predict=False):
    self.is_training = is_training 
    self.is_predict = is_predict

    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size()
    self.vocab_size = vocab_size
    
    self.end_id = vocabulary.end_id()
    self.get_start_id()
    assert self.end_id != vocabulary.vocab.unk_id(), 'input vocab generated without end id'

    self.emb_dim = emb_dim = FLAGS.emb_dim

    #--- for perf problem here exchange w_t and w https://github.com/tensorflow/tensorflow/issues/4138
    self.num_units = num_units = FLAGS.rnn_hidden_size
    with tf.variable_scope('output_projection'):
      self.w_t = melt.variable.get_weights_truncated('w', 
                                             [vocab_size, num_units], 
                                             stddev=FLAGS.weight_stddev) 
      self.w = tf.transpose(self.w_t)
      self.v = melt.variable.get_weights_truncated('v', 
                                             [vocab_size], 
                                             stddev=FLAGS.weight_stddev) 

    self.cell = melt.create_rnn_cell( 
      num_units=num_units,
      is_training=is_training, 
      keep_prob=FLAGS.keep_prob, 
      num_layers=FLAGS.num_layers, 
      cell_type=FLAGS.cell)

    num_sampled = FLAGS.num_sampled if not (is_predict and FLAGS.predict_no_sample) else 0
    self.softmax_loss_function = self.gen_sampled_softmax_loss_function(num_sampled)

  #for show and tell input is image input feature
  #for textsum input is None, sequence will pad <GO> 
  def sequence_loss(self, input, sequence, initial_state=None, emb=None):
    if emb is None:
      emb = self.emb
    
    is_training = self.is_training
    batch_size = tf.shape(sequence)[0]

    sequence, sequence_length = melt.pad(sequence,
                                     start_id=self.get_start_id(),
                                     end_id=self.get_end_id())

    #TODO different init state as show in ptb_word_lm
    state = self.cell.zero_state(batch_size, tf.float32) if initial_state is None else initial_state

    #[batch_size, num_steps - 1, emb_dim], remove last col
    inputs = tf.nn.embedding_lookup(emb, melt.dynamic_exclude_last_col(sequence))
    
    if is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
    
    #inputs[batch_size, num_steps, emb_dim] input([batch_size, emb_dim] -> [batch_size, 1, emb_dim]) before concat
    if input is not None:
      #used like showandtell where image_emb is as input, additional to sequence
      inputs = tf.concat(1, [tf.expand_dims(input, 1), inputs])
    else:
      #common usage input is None, sequence as input, notice already pad <GO> before using melt.pad
      sequence_length -= 1
      sequence = sequence[:, 1:]
    
    if self.is_predict:
      #---only need when predict, since train input already dynamic length, NOTICE this will improve speed a lot
      num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
      sequence = tf.slice(sequence, [0,0], [-1, num_steps])

    outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, 
                                       initial_state=state, 
                                       sequence_length=sequence_length)
    self.final_state = state
    
    if self.softmax_loss_function is None:
      #[batch_size, num_steps, num_units] * [num_units, vocab_size] -> logits [batch_size, num_steps, vocab_size]
      logits = melt.batch_matmul_embedding(outputs, self.w) + self.v
    else:
      logits = outputs

    #[batch_size, num_steps]
    targets = sequence
    mask = tf.cast(tf.sign(sequence), dtype=tf.float32)
    
    if self.is_predict and FLAGS.predict_no_sample:
      loss = melt.seq2seq.exact_predict_loss(logits, batch_size, num_steps)
    else:
      #loss [batch_size,] 
      loss = melt.seq2seq.sequence_loss_by_example(
          logits,
          targets,
          weights=mask,
          softmax_loss_function=self.softmax_loss_function)
    
    #mainly for compat with [bach_size, num_losses]
    loss = tf.reshape(loss, [-1, 1])
    if self.is_predict:
      loss = self.normalize_length(loss, sequence_length)
 
    return loss

  #TODO better, handle, without encoder_output?
  def get_start_input(self, encoder_output):
    batch_size = tf.shape(encoder_output)[0]
    start_input = melt.constants(self.start_id, [batch_size], tf.int32)
    return start_input

  def get_start_embedding_input(self, encoder_output, emb=None):
    if emb is None:
      emb = self.emb
    start_input = self.get_start_input(encoder_output)
    start_embedding_input = tf.nn.embedding_lookup(emb, start_input) 
    return start_embedding_input

  def generate_sequence(self, input, max_steps, initial_state=None, 
                        decode_method=SeqDecodeMethod.max_prob, convert_unk=True, 
                        emb=None):
    if emb is None:
      emb = self.emb

    batch_size = tf.shape(input)[0]
    state = self.cell.zero_state(batch_size, tf.float32) if initial_state is None else initial_state
 
    generated_sequence = []

    last_symbol = input
    with tf.variable_scope("RNN"):
      for i in range(max_steps):
        #with tf.variable_scope("RNN", reuse=True if i > 0 else None):
        #https://github.com/tensorflow/tensorflow/issues/537
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        (output, state) = self.cell(last_symbol, state)

        logit_symbols = tf.nn.xw_plus_b(output, self.w, self.v)

        top_prob_symbols = None
        if decode_method == SeqDecodeMethod.max_prob:
          if not convert_unk:
            max_prob_symbol = tf.argmax(logit_symbols, 1)
          else:
            max_prob_symbols = tf.nn.top_k(logit_symbols, 2)[1]
            first_col_mask = tf.abs(tf.sign(max_prob_symbols[:,0] - vocabulary.vocab.unk_id()))
            first_col = max_prob_symbols[:, 0] * first_col_mask
            second_col_mask = 1 - first_col_mask
            second_col = max_prob_symbols[:, 1] * second_col_mask
            max_prob_symbol = first_col + second_col
        elif decode_method == SeqDecodeMethod.sample:
          max_prob_symbol = tf.nn.top_k(logit_symbols, beam_size)[1][:, np.random.choice(beam_size, 1)]
        elif decode_method == SeqDecodeMethod.full_sample:
          top_prob_symbols = tf.nn.top_k(logit_symbols, beam_size)[1]
          max_prob_symbol = top_prob_symbols[:, np.random.choice(beam_size, 1)]
        elif decode_method == SeqDecodeMethod.beam_search:
          raise ValueError('beam search nor implemented yet')
        else:
          raise ValueError('not supported decode method')

        #@TODO stop when seeing end id! ? now can not since batch end postion different, need c++ implement?
        last_symbol = tf.nn.embedding_lookup(emb, max_prob_symbol)
        max_prob_symbol = tf.reshape(max_prob_symbol, [batch_size, -1])

        if top_prob_symbols is not None:
          generated_sequence.append(top_prob_symbols)
        else:
          generated_sequence.append(max_prob_symbol)

      generated_sequence = tf.concat(1, generated_sequence)
      return generated_sequence, tf.zeros([batch_size,])

  def gen_sampled_softmax_loss_function(self, num_sampled):
    #@TODO move to melt  def prepare_sampled_softmax_loss(num_sampled, vocab_size, emb_dim)
    #return output_projection, softmax_loss_function
    #also consider candidate sampler 
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    # FIMXE seems not work when using meta graph load predic cause segmentaion fault if add not is_predict
    #if not is_predict and (num_sampled > 0 and num_sampled < vocab_size):
    vocab_size = self.vocab_size
    if num_sampled > 0 and num_sampled < vocab_size:
      def sampled_loss(inputs, labels):
        #with tf.device("/cpu:0"):
        #----move below to melt.seq2seq
        #labels = tf.reshape(labels, [-1, 1])

        if FLAGS.log_uniform_sample:
          #most likely go here
          sampled_values = None
          if self.is_predict:
            sampled_values = tf.nn.log_uniform_candidate_sampler(
              true_classes=labels,
              num_true=1,
              num_sampled=num_sampled,
              unique=True,
              range_max=vocab_size,
              seed=FLAGS.predict_sample_seed)
        else:
          vocab_counts_list = [vocabulary.vocab.freq(i) for i in xrange(vocab_size)]
          vocab_counts_list[0] = 1 #--hack..
          #vocab_counts_list = [vocabulary.vocab.freq(i) for i in xrange(NUM_RESERVED_IDS, vocab_size)
          #-------above is ok, do not go here
          #@TODO find which is better log uniform or unigram sample, what's the diff with full vocab
          #num_reserved_ids: Optionally some reserved IDs can be added in the range
          #`[0, num_reserved_ids]` by the users. One use case is that a special
          #unknown word token is used as ID 0. These IDs will have a sampling
          #probability of 0.
          #@TODO seems juse set num_reserved_ids to 1 to ignore pad, will we need to set to 2 to also ignore UNK?
          #range_max: An `int` that is `>= 1`.
          #The sampler will sample integers from the interval [0, range_max).

          ##---@FIXME setting num_reserved_ids=NUM_RESERVED_IDS will cause Nan, why...
          #sampled_values = tf.nn.fixed_unigram_candidate_sampler(
          #    true_classes=labels,
          #    num_true=1,
          #    num_sampled=num_sampled,
          #    unique=True,
          #    range_max=vocab_size - NUM_RESERVED_IDS,
          #    distortion=0.75,
          #    num_reserved_ids=NUM_RESERVED_IDS,
          #    unigrams=vocab_counts_list)
          
          #FIXME keyword app, now tested on basice 50w,
          # sample 1000 or other num will cause nan use sampled by count or not by count
          #flickr data is ok
          # IMPORTANT to find out reason, data dependent right now
          #seems ok if using dynamic_batch=1, but seems no speed gain... only 5 instance/s
          #NOTICE 50w keyword very slow with sample(1w) not speed gain
          #4GPU will outof mem FIXME  https://github.com/tensorflow/tensorflow/pull/4270
          #https://github.com/tensorflow/tensorflow/issues/4138
          sampled_values = tf.nn.fixed_unigram_candidate_sampler(
              true_classes=labels,
              num_true=1,
              num_sampled=num_sampled,
              unique=True,
              range_max=vocab_size,
              distortion=0.75,
              num_reserved_ids=0,
              unigrams=vocab_counts_list)

        return tf.nn.sampled_softmax_loss(self.w_t, 
                                          self.v, 
                                          labels=labels, 
                                          inputs=inputs, 
                                          num_sampled=num_sampled, 
                                          num_classes=vocab_size,
                                          sampled_values=sampled_values)
                                 
      return sampled_loss
    else:
      return None

  def normalize_length(self, loss, sequence_length):
    if not FLAGS.predict_no_sample:
      sequence_length = tf.cast(sequence_length, tf.float32)
      sequence_length = tf.reshape(sequence_length, [-1, 1])
      loss = loss * sequence_length 
    normalize_factor = tf.pow(sequence_length, FLAGS.length_normalization_factor)
    loss /= normalize_factor  
    return loss

  def get_start_id(self):
    start_id = None 
    if not FLAGS.input_with_start_mark and FLAGS.add_text_start:
      if FLAGS.zero_as_text_start:
        start_id = 0
      elif FLAGS.go_as_text_start:
        start_id = vocabulary.go_id()
      else:
        start_id = vocabulary.start_id()
    self.start_id = start_id
    return start_id

  def get_end_id(self):
    if (FLAGS.input_with_end_mark or (self.is_predict and not FLAGS.predict_with_end_mark)):
      return None 
    else:
      return self.end_id


  def generate_sequence_by_beam_search(self, input, max_steps, initial_state=None, beam_size=5, 
                                       convert_unk=True, length_normalization_factor=1.0, emb=None):
    if emb is None:
      emb = self.emb

    def loop_function(prev, i):
      logit_symbols = tf.nn.embedding_lookup(emb, prev)
      return logit_symbols

    state = self.cell.zero_state(tf.shape(input)[0], tf.float32) if initial_state is None else initial_state
    
    return melt.seq2seq.beam_decode(input, max_steps, state, 
                                    self.cell, loop_function, scope="RNN",
                                    beam_size=beam_size, done_token=vocabulary.vocab.end_id(), 
                                    output_projection=(self.w, self.v),
                                    length_normalization_factor=length_normalization_factor,
                                    topn=10)