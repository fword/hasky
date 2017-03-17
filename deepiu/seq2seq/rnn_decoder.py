#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_decoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:05.991481
#   \Description  
# ==============================================================================
"""
The diff for this RnnDecoder with google im2txt is im2txt assume a start word after image_embedding
so by doing this im2text or textsum all be the same input is embedding of <start_id> and inital_state
the first lstm_cell(image_embedding) outputs is discarded just use output_state as inital_state with <start_id>
so decode process always assume to first start with <start_id>

here we assume showandtell not add this <start_id> by default so image_embedding output assume to be first
decoded word, image_embedding is the first input ans inital_state is zero_state

#so not check if ok add additinal text start as first input before image...

and decode process always assume to have first_input and inital_state as input

TODO decode functions also accept first_input=None so just treat as im2txt
by this way general seq2seq also simpler as interactive beam decode start from start_id 
(input can start from start_id outside graph, not as right now must fetch from ingraph...
for like image_embedding as input you can not get it from lookup from embedding)  may be better(simpler) design
but one more train decode step, but since no additional decoding calc softmax not much perf hurt
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_sampled', 10000, 'num samples of neg word from vocab')
flags.DEFINE_boolean('log_uniform_sample', True, '')

flags.DEFINE_boolean('add_text_start', False, """if True will add <s> or 0 or GO before text 
                                              as first input before image, by default will be GO, 
                                              make sure add_text_start==True if you use seq2seq""")
flags.DEFINE_boolean('zero_as_text_start', False, """if add_text_start, 
                                                    here True will add 0 False will add <s>
                                                    0 means the loss from image to 0/pad not considered""")
flags.DEFINE_boolean('go_as_text_start', True, """ """)

flags.DEFINE_boolean('input_with_start_mark', False, """if input has already with <S> start mark""")
flags.DEFINE_boolean('input_with_end_mark', False, """if input has already with </S> end mark""")

flags.DEFINE_boolean('predict_with_end_mark', True, """if predict with </S> end mark""")

flags.DEFINE_float('length_normalization_factor', 0., """If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.  see tensorflow/models/im2text
        by default wil follow im2text set to 0""")

flags.DEFINE_boolean('predict_use_prob', True, 'if True then use exp(logprob) and False will direclty output logprob')
flags.DEFINE_boolean('predict_no_sample', False, 'if True will use exact loss')
flags.DEFINE_integer('predict_sample_seed', 0, '')

flags.DEFINE_boolean('use_attention', False, 'wether to use attention for decoder')
flags.DEFINE_string('attention_option', 'luong', 'luong or bahdanau')

flags.DEFINE_integer('beam_size', 10, 'for seq decode beam search size')
flags.DEFINE_integer('decode_max_words', 0, 'if 0 use TEXT_MAX_WORDS from conf.py otherwise use decode_max_words')

import functools
import melt 
from deepiu.util import vocabulary
from deepiu.seq2seq.decoder import Decoder

class SeqDecodeMethod:
  greedy = 0
  sample = 1
  full_sample = 2
  beam = 3         # ingraph beam search
  beam_search = 4  # outgraph beam search/ interactve beam search

class RnnDecoder(Decoder):
  def __init__(self, is_training=True, is_predict=False):
    self.scope = 'rnn'
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
      #weights
      self.w = tf.transpose(self.w_t)
      #biases
      self.v = melt.variable.get_weights_truncated('v', 
                                             [vocab_size], 
                                             stddev=FLAGS.weight_stddev) 

    #TODO https://github.com/tensorflow/tensorflow/issues/6761  tf 1.0 will fail if not scope='rnn' the same as when using self.cell...
   
    self.cell = melt.create_rnn_cell( 
      num_units=num_units,
      is_training=is_training, 
      keep_prob=FLAGS.keep_prob, 
      num_layers=FLAGS.num_layers, 
      cell_type=FLAGS.cell)

    num_sampled = FLAGS.num_sampled if not (is_predict and FLAGS.predict_no_sample) else 0
    self.softmax_loss_function = melt.seq2seq.gen_sampled_softmax_loss_function(num_sampled, 
                                                                                self.vocab_size, 
                                                                                self.w_t,
                                                                                self.v,
                                                                                FLAGS.log_uniform_sample,
                                                                                is_predict=self.is_predict,
                                                                                sample_seed=FLAGS.predict_sample_seed,
                                                                                vocabulary=vocabulary)
    
  def sequence_loss(self, input, sequence, 
                    initial_state=None, attention_states=None, 
                    exact_prob=False, exact_loss=False,
                    emb=None):
    """
    for general seq2seq input is None, sequence will pad <GO>, inital_state is last state from encoder
    for img2text/showandtell input is image_embedding, inital_state is None/zero set
    TODO since exact_porb and exact_loss same value, may remove exact_prob
    """
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
      inputs = tf.concat([tf.expand_dims(input, 1), inputs], 1)
    else:
      #common usage input is None, sequence as input, notice already pad <GO> before using melt.pad
      sequence_length -= 1
      sequence = sequence[:, 1:]
    
    if self.is_predict:
      #---only need when predict, since train input already dynamic length, NOTICE this will improve speed a lot
      num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
      sequence = sequence[:, 0:num_steps]
      inputs = inputs[:, 0:num_steps, :]

    if attention_states is None:
      outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, 
                                       initial_state=state, 
                                       sequence_length=sequence_length,
                                       scope=self.scope)
      self.final_state = state
    else:
      attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
        self.prepare_attention(attention_states)
      decoder_fn_train = melt.seq2seq.attention_decoder_fn_train(
          encoder_state=state,
          attention_keys=attention_keys,
          attention_values=attention_values,
          attention_score_fn=attention_score_fn,
          attention_construct_fn=attention_construct_fn)
      decoder_outputs_train, decoder_state_train, _ = \
                    melt.seq2seq.dynamic_rnn_decoder(
                        cell=self.cell,
                        decoder_fn=decoder_fn_train,
                        inputs=inputs,
                        sequence_length=tf.cast(sequence_length, tf.int32),
                        scope=self.scope)
      outputs = decoder_outputs_train

      self.final_state = decoder_state_train
    
    #TODO: hack here add FLAGS.predict_no_sample just for Seq2seqPredictor exact_predict
    softmax_loss_function = self.softmax_loss_function
    if self.is_predict and (exact_prob or exact_loss):
      softmax_loss_function = None

    #[batch_size, num_steps]
    targets = sequence
    
    if softmax_loss_function is None:
      #[batch_size, num_steps, num_units] * [num_units, vocab_size]
      # -> logits [batch_size, num_steps, vocab_size] (if use exact_predict_loss)
      #or [batch_size * num_steps, vocab_size] by default flatten=True
      keep_dims = exact_prob or exact_loss
      logits = melt.batch_matmul_embedding(outputs, self.w, keep_dims=keep_dims) + self.v
      if not keep_dims:
        targets = tf.reshape(targets, [-1])
    else:
      logits = outputs

    mask = tf.cast(tf.sign(sequence), dtype=tf.float32)
    
    if self.is_predict and exact_prob:
      #generate real prob for sequence
      #for 10w vocab textsum seq2seq 20 -> 4 about 
      loss = melt.seq2seq.exact_predict_loss(logits, targets, mask, num_steps, batch_size)
    elif self.is_predict and exact_loss: 
      #force no sample softmax loss, the diff with exact_prob is here we just use cross entropy error as result not real prob of seq
      #NOTICE using time a bit less  55 to 57(prob), same result with exact prob and exact score
      #but 256 vocab sample will use only about 10ms
      #TODO check more with softmax loss and sampled somtmax loss, check length normalize
      loss = melt.seq2seq.sequence_loss_by_example(logits, targets, weights=mask)
    else:
      #loss [batch_size,] 
      loss = melt.seq2seq.sequence_loss_by_example(
          logits,
          targets,
          weights=mask,
          softmax_loss_function=softmax_loss_function)
    
    #mainly for compat with [bach_size, num_losses]
    loss = tf.reshape(loss, [-1, 1])

    if self.is_predict:
      loss = self.normalize_length(loss, sequence_length, exact_prob)
    return loss

  def generate_sequence_greedy(self, input, max_words, 
                        initial_state=None, attention_states=None,
                        convert_unk=True, 
                        emb=None):
    """
    this one is using greedy search method
    for beam search using generate_sequence_by_beam_search with addditional params like beam_size
    """
    if emb is None:
      emb = self.emb

    batch_size = tf.shape(input)[0]
    state = self.cell.zero_state(batch_size, tf.float32) if initial_state is None else initial_state
    
    def output_fn(output):
      return tf.nn.xw_plus_b(output, self.w, self.v)

    if attention_states is None:
      decoder_fn_inference = melt.seq2seq.greedy_decoder_fn_inference(
                    output_fn=output_fn,
                    first_input=input,
                    encoder_state=state,
                    embeddings=emb,
                    end_of_sequence_id=self.end_id,
                    maximum_length=max_words,
                    num_decoder_symbols=self.vocab_size,
                    dtype=tf.int32)
    else:
      #dynamic loop do not need to consider scope problem since attention construct will only do once
      attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
        self.prepare_attention(attention_states)
      decoder_fn_inference = (
          melt.seq2seq.attention_decoder_fn_inference(
              output_fn=output_fn,
              first_input=input,
              encoder_state=state,
              attention_keys=attention_keys,
              attention_values=attention_values,
              attention_score_fn=attention_score_fn,
              attention_construct_fn=attention_construct_fn,
              embeddings=emb,
              end_of_sequence_id=self.end_id,
              maximum_length=max_words,
              num_decoder_symbols=self.vocab_size,
              dtype=tf.int32))

    decoder_outputs_inference, decoder_state_inference, decoder_context_state_inference = \
      melt.seq2seq.dynamic_rnn_decoder(
               cell=self.cell,
               decoder_fn=decoder_fn_inference,
               scope=self.scope)
    
    generated_sequence = tf.transpose(decoder_context_state_inference.stack(), [1, 0])

    #------like beam search return sequence, score
    return generated_sequence, tf.zeros([batch_size,])

  def generate_sequence_beam(self, input, max_words, 
                             initial_state=None, attention_states=None,
                             beam_size=5, convert_unk=True,
                             length_normalization_factor=0., 
                             emb=None):
    """
    beam dcode means ingraph beam search
    return top (path, score)
    """
    if emb is None:
      emb = self.emb

    def loop_function(i, prev, state, decoder):
      prev, state, attention = decoder.take_step(i, prev, state)

      logit_symbols = tf.nn.embedding_lookup(emb, prev)
      if attention is not None:
        logit_symbols = tf.concat([logit_symbols, attention], 1)

      #logit_symbols is next input
      return logit_symbols, state

    state = self.cell.zero_state(tf.shape(input)[0], tf.float32) if initial_state is None else initial_state
    
    attention_keys, attention_values, attention_score_fn, attention_construct_fn = None, None, None, None
    if attention_states is not None:
      attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
        self.prepare_attention(attention_states)
        
    #TODO to be safe make topn the same as beam size
    return melt.seq2seq.beam_decode(input, max_words, state, 
                                    self.cell, loop_function, scope=self.scope,
                                    beam_size=beam_size, done_token=vocabulary.vocab.end_id(), 
                                    output_projection=(self.w, self.v),
                                    length_normalization_factor=length_normalization_factor,
                                    topn=beam_size, 
                                    #topn=1,
                                    attention_construct_fn=attention_construct_fn,
                                    attention_keys=attention_keys,
                                    attention_values=attention_values)    

  def generate_sequence_beam_search(self, input, max_words=None, 
                                  initial_state=None, attention_states=None,
                                  beam_size=10, convert_unk=True,
                                  length_normalization_factor=0., 
                                  emb=None):
    """
    outgraph beam search, input should be one instance only batch_size=1
    max_words actuaolly not used here... for it is determined outgraph..
    return top (path, score)
    TODO add attention support!
    """
    if emb is None:
      emb = self.emb

    if attention_states is not None:
       attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
        self.prepare_attention(attention_states)
    else:
       attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
        None, None, None, None

    beam_search_step = functools.partial(self.beam_search_step, 
                                         beam_size=beam_size,
                                         attention_construct_fn=attention_construct_fn,
                                         attention_keys=attention_keys,
                                         attention_values=attention_values)

    #since before hack using generate_sequence_greedy, here can not set scope.reuse_variables
    #NOTICE inorder to use lstm which is in .../rnn/ nameapce here you must also add this scope to use the shared 
    with tf.variable_scope(self.scope) as scope:
      if attention_states is not None:
        inital_attention = melt.seq2seq.init_attention(initial_state)
        input = tf.concat([input, inital_attention], 1)

      inital_attention, initial_state, initial_logprobs, initial_ids = beam_search_step(input, initial_state)

      scope.reuse_variables()
      # In inference mode, use concatenated states for convenient feeding and
      # fetching.
      state_is_tuple = len(initial_state) == 2

      if state_is_tuple:
        initial_state = tf.concat(initial_state, 1, name="initial_state")
        state_size = sum(self.cell.state_size)
      else:
        state_size = self.cell.state_size

      #output is used only when use attention
      if attention_states is not None:
        initial_state = tf.concat([initial_state, inital_attention], 1, name="initial_attention_state")
        state_size += self.cell.output_size

      tf.add_to_collection('beam_search_initial_state', initial_state)
      tf.add_to_collection('beam_search_initial_logprobs', initial_logprobs)
      tf.add_to_collection('beam_search_initial_ids', initial_ids)
      tf.add_to_collection('beam_search_beam_size', tf.constant(beam_size))

      input_feed = tf.placeholder(dtype=tf.int64, 
                                  shape=[None],  # batch_size
                                  name="input_feed")
      tf.add_to_collection('beam_search_input_feed', input_feed)
      input = tf.nn.embedding_lookup(emb, input_feed)


      # Placeholder for feeding a batch of concatenated states.
      state_feed = tf.placeholder(dtype=tf.float32,
                                  shape=[None, state_size],
                                  name="state_feed")
      tf.add_to_collection('beam_search_state_feed', state_feed)

      if attention_states is not None:
        state, attention = tf.split(state_feed, [state_size - self.cell.output_size, self.cell.output_size], axis=1)
      else:
        state = state_feed

      if state_is_tuple:
        state = tf.split(state, num_or_size_splits=2, axis=1)

      if attention_states is not None:
        input = tf.concat([input, attention], 1)
      
      attention, state, top_logprobs, top_ids = beam_search_step(input, state)

      if state_is_tuple:
        # Concatentate the resulting state.
        state = tf.concat(state, 1, name="state")
      if attention_states is not None:
        state = tf.concat([state, attention], 1, name="attention_state")

      tf.add_to_collection('beam_search_state', state)
      tf.add_to_collection('beam_search_logprobs', top_logprobs)
      tf.add_to_collection('beam_search_ids', top_ids)

      #just same return like return path list, score list
      return tf.no_op(), tf.no_op()

  def beam_search_step(self, input, state, beam_size, 
                       attention_construct_fn=None,
                       attention_keys=None,
                       attention_values=None):
    output, state = self.cell(input, state)

    #TODO: this step cause.. attenion decode each step after initalization still need input_text feed 
    #will this case attention_keys and attention_values to be recompute(means redo encoding process) each step?
    #can we avoid this? seems not better method, 
    #if enocding is slow may be feed attention_keys, attention_values each step
    if attention_construct_fn is not None:
      output = attention_construct_fn(output, attention_keys, attention_values)
    
    logits = tf.nn.xw_plus_b(output, self.w, self.v)
    logprobs = tf.nn.log_softmax(logits)

    top_logprobs, top_ids = tf.nn.top_k(logprobs, beam_size)

    return output, state, top_logprobs, top_ids

  def prepare_attention(self, attention_states):
    decoder_hidden_size = self.num_units
    attention_option = FLAGS.attention_option  # can be "bahdanau"
    print('attention_option:', attention_option)
    assert attention_option is "luong" or attention_option is "bahdanau"
    attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
       melt.seq2seq.prepare_attention(attention_states, attention_option, self.num_units)
    return attention_keys, attention_values, attention_score_fn, attention_construct_fn

  def get_start_input(self, batch_size):
    start_input = melt.constants(self.start_id, [batch_size], tf.int32)
    return start_input

  def get_start_embedding_input(self, batch_size, emb=None):
    if emb is None:
      emb = self.emb
    start_input = self.get_start_input(batch_size)
    start_embedding_input = tf.nn.embedding_lookup(emb, start_input) 
    return start_embedding_input

  def normalize_length(self, loss, sequence_length, exact_prob=False):
    sequence_length = tf.cast(sequence_length, tf.float32)
    #NOTICE must check if shape ok, [?,1] / [?,] will get [?,?]
    sequence_length = tf.reshape(sequence_length, [-1, 1])
    #-- below is used only when using melt.seq2seq.loss.exact_predict_loss
    if not exact_prob:
      #use sequence_loss_by_example with default average_across_timesteps=True, so we just turn it back
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



