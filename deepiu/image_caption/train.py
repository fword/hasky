#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================
"""
@TODO using logging ?
NOTICE using feed_dict --feed_neg=1 will slow *10
@TODO why evaluate will use mem up to 13% then down to 5%?
using tf.nn.top_k do sort in graph still evaluate large mem same as before
20w will using 1.6G 32 * 5%
50w 9.1%
200w 16G?
2000w 160g?

@TODO why using gpu will use more cpu mem?
50w keyword MAX_TEXT_WORDS 20
cpu version: 
show_eval=0  train 6.7% eval 11.4%
show_eval=1 train 8% eval 20%

gpu version: 
show_eval=0 train 8% eval 12.8%
show_eval=1 train 35% eval 39%

text means text ids
text_str means orit text str
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './model', '')

flags.DEFINE_string('algo', 'bow', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')

flags.DEFINE_boolean('debug', False, '')

import sys
import functools
import gezi
import melt
logging = melt.logging

from deepiu.image_caption import input_app as InputApp

from deepiu.image_caption import eval_show
from deepiu.image_caption import evaluator

from deepiu.image_caption.algos import algos_factory

#debug
from deepiu.util import vocabulary
from deepiu.util import text2ids

from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

sess = None

#feed ops maninly for inital state re feed, see ptb_word_lm.py, notice will be much slower adding this
# state = m.initial_state.eval()
# for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
#                                                   m.num_steps)):
#   cost, state, _ = session.run([m.cost, m.final_state, eval_op],
#                                {m.input_data: x,
#                                 m.targets: y,
#                                 m.initial_state: state})
feed_ops = []
feed_run_ops = []
feed_results = []

#TODO do not consider feed dict support right now for tower loss!
def tower_loss(trainer, input_app=None, input_results=None):
  if input_app is None:
    input_app = InputApp.InputApp()
  if input_results is None:
    input_results = input_app.gen_input(train_only=True)

  #--------train
  image_name, image_feature, text, text_str = input_results[input_app.input_train_name]
  #--------train neg
  if input_app.input_train_neg_name in input_results:
    neg_text, neg_text_str = input_results[input_app.input_train_neg_name]
  else:
    neg_text, neg_text_str = None, None

  loss = trainer.build_train_graph(image_feature, text, neg_text)
  return loss

def gen_train_graph(input_app, input_results, trainer):
  """
    main flow, key graph
  """
  #--- if you don't want to use mutli gpu, here just for safe(code same with old single gpu cod)
  if FLAGS.num_gpus == 0:
    loss = tower_loss(trainer, input_app, input_results)
  else:
    loss_function = lambda: tower_loss(trainer)
    #here loss is a list of losses
    loss = melt.tower_losses(loss_function, FLAGS.num_gpus)
    print('num tower losses:', len(loss))

  ops = [loss]
  #--------mark train graph finished, all graph after must share variable from train graph
  #melt.reuse_variables()
  trainer.is_training = False
    
  deal_debug_results = None
  if FLAGS.debug == True:
    ops += [tf.get_collection('scores')[-1], tf.get_collection('text_feature')[-1]]
    
    def _deal_debug_results(results):
      _, scores, text_feature = results
      print('scores', scores)
      print('text_feature', text_feature)

    # if not FLAGS.feed_dict:
    #   ops += [text, text_str, neg_text, neg_text_str]

    # def _deal_debug_results(results):
    #   if FLAGS.feed_dict:
    #     _, scores = results
    #   else:
    #     _, scores, text, text_str, neg_text, neg_text_str = results
    #   print(scores)
    #   if not FLAGS.feed_dict:
    #     print(text_str[0], text[0], text2ids.ids2text(text[0]))
    #     print(neg_text_str[0][0][0], neg_text[0][0], text2ids.ids2text(neg_text[0][0]))

    #     # global step
    #     # if step == 42:
    #     #   print(neg_text_str[8][3][0], neg_text[8][3], text2ids.ids2text(neg_text[8][3])) 
    #     # step += 1

    deal_debug_results = _deal_debug_results

    ###----------show how to debug
    #debug_ops = [text, neg_text, trainer.emb, trainer.scores]
    #debug_ops += trainer.gradients
    #print(trainer.gradients)
    #ops += debug_ops
    #def _deal_debug_results(results):
    #  for result in results[-len(debug_ops):]:
    #    #print(result.shape)
    #    print(result)
    #deal_debug_results = _deal_debug_results

  return ops, deal_debug_results

def gen_train(input_app, input_results, trainer):
  input_gen_feed_dict = input_app.gen_feed_dict
  
  ops, deal_debug_results = gen_train_graph(input_app, input_results, trainer)
  
  #@NOTICE make sure global feed dict ops put at last
  if hasattr(trainer, 'feed_ops'):
    global feed_ops, feed_run_ops, feed_results
    feed_ops, feed_run_ops = trainer.feed_ops()
    #feed_results = sess.run(feed_run_ops)
    ops += feed_run_ops

  def _deal_results(results):
    melt.print_results(results, ['loss'])

    if deal_debug_results is not None:
      debug_results = results[:-len(feed_ops)] if feed_ops else results
      deal_debug_results(debug_results)

    if feed_ops:
      global feed_results
      feed_results = results[-len(feed_ops):]

  deal_results = _deal_results


  def _gen_feed_dict():
    feed_dict = input_gen_feed_dict()
    
    if feed_results:
      for op, result in zip(feed_ops, feed_results):
        feed_dict[op] = result
        #print(op, result)
    return feed_dict

  gen_feed_dict = _gen_feed_dict

  return ops, gen_feed_dict, deal_results

def gen_evalulate(input_app, 
                  input_results,
                  predictor, 
                  eval_ops, 
                  eval_scores,
                  eval_neg_text=None,
                  eval_neg_text_str=None):
  if algos_factory.is_discriminant(FLAGS.algo):
    eval_ops += eval_show.gen_eval_show_ops(
        input_app, 
        input_results, 
        predictor, 
        eval_scores, 
        eval_neg_text, 
        eval_neg_text_str)
    deal_eval_results = eval_show.deal_eval_results
  else:
    eval_ops += eval_show.gen_eval_generated_texts_ops(
        input_app, 
        input_results, 
        predictor, 
        eval_scores,
        eval_neg_text,
        eval_neg_text_str)
    deal_eval_results = eval_show.deal_eval_generated_texts_results
  return eval_ops, deal_eval_results

def gen_validate(input_app, input_results, trainer, predictor):
  gen_eval_feed_dict = input_app.gen_eval_feed_dict
  eval_ops = None
  train_with_validation = input_results[input_app.input_valid_name] is not None
  deal_eval_results = None
  if train_with_validation and not FLAGS.train_only:
    eval_image_name, eval_image_feature, eval_text, eval_text_str = input_results[input_app.input_valid_name]
    if input_app.input_valid_neg_name in input_results:
      eval_neg_text, eval_neg_text_str = input_results[input_app.input_valid_neg_name]
    else:
      eval_neg_text, eval_neg_text_str = None, None

    eval_loss = trainer.build_train_graph(eval_image_feature, eval_text, eval_neg_text)
    eval_scores = tf.get_collection('scores')[-1]
    eval_ops = [eval_loss]

    if FLAGS.show_eval and (predictor is not None):
      eval_ops, deal_eval_results = \
        gen_evalulate(
            input_app, 
            input_results, 
            predictor, 
            eval_ops, 
            eval_scores,
            eval_neg_text,
            eval_neg_text_str)
    else:
      deal_eval_results = lambda x: melt.print_results(x, ['loss'])

  return eval_ops, gen_eval_feed_dict, deal_eval_results

def gen_predict_graph(predictor):  
  """
  call it at last , build predict graph
  the probelm here is you can not change like beam size later...
  """
  #-----discriminant and generative
  score = predictor.init_predict()
  tf.add_to_collection('score', score)

  if algos_factory.is_discriminant(FLAGS.algo):
    tf.add_to_collection('textsim_score', predictor.textsim_score)
  
   #-----generateive
  if algos_factory.is_generative(FLAGS.algo):
    exact_score = predictor.init_predict(exact_loss=True)
    tf.add_to_collection('exact_score', exact_score)

    init_predict_text = functools.partial(predictor.init_predict_text, 
                                          beam_size=FLAGS.beam_size, 
                                          convert_unk=False)
    text, text_score = init_predict_text(decode_method=FLAGS.seq_decode_method)
    beam_text, beam_text_score = init_predict_text(decode_method=SeqDecodeMethod.beam)

    tf.add_to_collection('text', text)
    tf.add_to_collection('text_score', text_score)
    tf.add_to_collection('beam_text', beam_text)
    tf.add_to_collection('beam_text_score', beam_text_score)

#step = 0
def train_process(trainer, predictor=None):
  input_app = InputApp.InputApp()
  input_results = input_app.gen_input()

  with tf.variable_scope('run') as scope:
    ops, gen_feed_dict, deal_results = gen_train(
     input_app, 
     input_results, 
     trainer)
  
    scope.reuse_variables()

    #saving predict graph, so later can direclty predict without building from scratch
    #also used in gen validate if you want to use direclty predict as evaluate per epoch
    if predictor is not None:
     gen_predict_graph(predictor)

    eval_ops, gen_eval_feed_dict, deal_eval_results = gen_validate(
      input_app, 
      input_results, 
      trainer, 
      predictor)

    metric_eval_function = None
    if FLAGS.metric_eval:
      #generative can do this also but it is slow so just ingore this
      if not algos_factory.is_generative(FLAGS.algo): 
        metric_eval_function = lambda: evaluator.evaluate_scores(predictor, random=True)

  melt.apps.train_flow(ops, 
                       gen_feed_dict=gen_feed_dict,
                       deal_results=deal_results,
                       eval_ops=eval_ops,
                       gen_eval_feed_dict=gen_eval_feed_dict,
                       deal_eval_results=deal_eval_results,
                       optimizer=FLAGS.optimizer,
                       learning_rate=FLAGS.learning_rate,
                       num_steps_per_epoch=input_app.num_steps_per_epoch,
                       model_dir=FLAGS.model_dir,
                       metric_eval_function=metric_eval_function,
                       sess=sess)#notice if use melt.constant in predictor then must pass sess
  
def train():
  trainer, predictor =  algos_factory.gen_trainer_and_predictor(FLAGS.algo)
  logging.info('trainer:{}'.format(trainer))
  logging.info('predictor:{}'.format(predictor))
  train_process(trainer, predictor)

def main(_):
  #-----------init global resource
  logging.set_logging_path(gezi.get_dir(FLAGS.model_dir))

  InputApp.init()
  vocabulary.init()
  text2ids.init()
  evaluator.init()

  logging.info('algo:{}'.format(FLAGS.algo))
  logging.info('monitor_level:{}'.format(FLAGS.monitor_level))
  
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
 
  global sess
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)
  with tf.variable_scope(global_scope):
    train()
 
if __name__ == '__main__':
  tf.app.run()
