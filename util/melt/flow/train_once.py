#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 12:59:29.331219
#   \Description  
# ==============================================================================

"""
@TODO better logging, using logging.info ?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os

from melt.utils import logging
#import logging

import tensorflow as tf

import gezi
from gezi import Timer, AvgScore 
import melt

projector_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

def train_once(sess, 
               step, 
               ops, 
               names=None,
               gen_feed_dict=None, 
               deal_results=melt.print_results, 
               interval_steps=100,
               eval_ops=None, 
               eval_names=None, 
               gen_eval_feed_dict=None, 
               deal_eval_results=melt.print_results, 
               eval_interval_steps=100, 
               print_time=True, 
               print_avg_loss=True, 
               model_dir=None, 
               log_dir=None, 
               is_start=False,
               num_steps_per_epoch=None,
               metric_eval_function=None,
               metric_eval_interval_steps=0):

  timer = gezi.Timer()
  if print_time:
    if not hasattr(train_once, 'timer'):
      train_once.timer = Timer()
      train_once.eval_timer = Timer()
      train_once.metric_eval_timer = Timer()
   
  melt.set_global('step', step)
  epoch = step / num_steps_per_epoch  if num_steps_per_epoch else -1
  epoch_str = 'epoch:%.4f'%(epoch) if num_steps_per_epoch else ''
  melt.set_global('epoch', '%.4f'%(epoch))
  
  info = BytesIO()
  stop = False

  if ops is not None:
    if deal_results is None and names is not None:
      deal_results = lambda x: melt.print_results(x, names)
    if deal_eval_results is None and eval_names is not None:
      deal_eval_results = lambda x: melt.print_results(x, eval_names)

    if eval_names is None:
      eval_names = names 

    feed_dict = {} if gen_feed_dict is None else gen_feed_dict()
    
    results = sess.run(ops, feed_dict=feed_dict) 

    # #--------trace debug
    # if step == 210:
    #   run_metadata = tf.RunMetadata()
    #   results = sess.run(
    #         ops,
    #         feed_dict=feed_dict,
    #         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #         run_metadata=run_metadata)
    #   from tensorflow.python.client import timeline
    #   trace = timeline.Timeline(step_stats=run_metadata.step_stats)

    #   trace_file = open('timeline.ctf.json', 'w')
    #   trace_file.write(trace.generate_chrome_trace_format())
    
    
    #reults[0] assume to be train_op
    results = results[1:]
    
    #@TODO should support aver loss and other avg evaluations like test..
    if print_avg_loss:
      if not hasattr(train_once, 'avg_loss'):
        train_once.avg_loss = AvgScore()
        if interval_steps != eval_interval_steps:
          train_once.avg_loss2 = AvgScore()
      #assume results[0] as train_op return, results[1] as loss
      loss = gezi.get_singles(results)
      train_once.avg_loss.add(loss) 
      if interval_steps != eval_interval_steps:
        train_once.avg_loss2.add(loss)
    
    if is_start or interval_steps and step % interval_steps == 0:
      train_average_loss = train_once.avg_loss.avg_score()
      if print_time:
        duration = timer.elapsed()
        duration_str = 'duration:{:.3f} '.format(duration)
        melt.set_global('duration', '%.3f'%duration)
        info.write(duration_str)
        elapsed = train_once.timer.elapsed()
        steps_per_second = interval_steps / elapsed
        batch_size = melt.batch_size()
        num_gpus = melt.num_gpus()
        instances_per_second = interval_steps * batch_size * num_gpus / elapsed
        if num_gpus == 1:
          info.write('elapsed:[{:.3f}] batch_size:[{}] batches/s:[{:.2f}] insts/s:[{:.2f}] '.format(elapsed, batch_size, steps_per_second, instances_per_second))
        else:
          info.write('elapsed:[{:.3f}] batch_size:[{}] gpus:[{}], batches/s:[{:.2f}] insts/s:[{:.2f}] '.format(elapsed, batch_size, num_gpus, steps_per_second, instances_per_second))

      if print_avg_loss:
        #info.write('train_avg_metrics:{} '.format(melt.value_name_list_str(train_average_loss, names)))
        names_ = melt.adjust_names(train_average_loss, names)
        info.write('train_avg_metrics:{} '.format(melt.parse_results(train_average_loss, names_)))
        #info.write('train_avg_loss: {} '.format(train_average_loss))
      
      #print(gezi.now_time(), epoch_str, 'train_step:%d'%step, info.getvalue(), end=' ') 
      logging.info2('{} {} {}'.format(epoch_str, 'train_step:%d'%step, info.getvalue()))
      
      if deal_results is not None:
        stop = deal_results(results)
  
  metric_evaluate = False
  # if metric_eval_function is not None \
  #   and ( (is_start and (step or ops is None))\
  #     or (step and ((num_steps_per_epoch and step % num_steps_per_epoch == 0) \
  #            or (metric_eval_interval_steps \
  #                and step % metric_eval_interval_steps == 0)))):
  #     metric_evaluate = True 
  if metric_eval_function is not None \
    and (is_start \
      or (num_steps_per_epoch and step % num_steps_per_epoch == 0) \
           or (metric_eval_interval_steps \
               and step % metric_eval_interval_steps == 0)):
    metric_evaluate = True
  
  if metric_evaluate:
     evaluate_results, evaluate_names = metric_eval_function()

  if is_start or eval_interval_steps and step % eval_interval_steps == 0:
    if ops is not None:
      if interval_steps != eval_interval_steps:
        train_average_loss = train_once.avg_loss2.avg_score()
      
      info = BytesIO()
      
      names_ = melt.adjust_names(results, names)

      train_average_loss_str = ''
      if print_avg_loss and interval_steps != eval_interval_steps:
        train_average_loss_str = melt.value_name_list_str(train_average_loss, names_)
        melt.set_global('train_loss', train_average_loss_str)
        train_average_loss_str = 'train_avg_loss:{} '.format(train_average_loss_str)

      if interval_steps != eval_interval_steps:
        #end = '' if eval_ops is None else '\n'
        #print(gezi.now_time(), epoch_str, 'eval_step: %d'%step, train_average_loss_str, end=end)
        logging.info2('{} eval_step: {} {}'.format(epoch_str, step, train_average_loss_str))
    
    if eval_ops is not None:
      eval_feed_dict = {} if gen_eval_feed_dict is None else gen_eval_feed_dict()
      #eval_feed_dict.update(feed_dict)
      
      #------show how to perf debug
      ##timer_ = gezi.Timer('sess run generate')
      ##sess.run(eval_ops[-2], feed_dict=None)
      ##timer_.print()
      
      timer_ = gezi.Timer('sess run eval_ops')
      eval_results = sess.run(eval_ops, feed_dict=eval_feed_dict)
      timer_.print()
      if deal_eval_results is not None:
        #@TODO user print should also use logging as a must ?
        #print(gezi.now_time(), epoch_str, 'eval_step: %d'%step, 'eval_metrics:', end='')
        logging.info2('{} eval_step: {} eval_metrics:'.format(epoch_str, step))
        eval_stop = deal_eval_results(eval_results)

      eval_loss = gezi.get_singles(eval_results)
      assert len(eval_loss) > 0
      if eval_stop is True: stop = True
      eval_names_ = melt.adjust_names(eval_loss, eval_names)

      melt.set_global('eval_loss', melt.parse_results(eval_loss, eval_names_))
    elif interval_steps != eval_interval_steps:
      #print()
      pass

    if log_dir:
      #timer_ = gezi.Timer('witting log')
      
      if not hasattr(train_once, 'summary_op'):
        try:
          train_once.summary_op = tf.summary.merge_all()
        except Exception:
          train_once.summary_op = tf.merge_all_summaries()

        melt.print_summary_ops()

        try:
          train_once.summary_train_op = tf.summary.merge_all(key=melt.MonitorKeys.TRAIN)
          train_once.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        except Exception:
          train_once.summary_train_op = tf.merge_all_summaries(key=melt.MonitorKeys.TRAIN)
          train_once.summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_once.summary_writer, projector_config)

      summary = tf.Summary()
      #so the strategy is on eval_interval_steps, if has eval dataset, then tensorboard evluate on eval dataset
      #if not have eval dataset, will evaluate on trainset, but if has eval dataset we will also monitor train loss
      if train_once.summary_train_op is not None:
        summary_str = sess.run(train_once.summary_train_op, feed_dict=feed_dict)
        train_once.summary_writer.add_summary(summary_str, step)

      if eval_ops is None:
        #get train loss, for every batch train
        if train_once.summary_op is not None:
          #timer2 = gezi.Timer('sess run')
          summary_str = sess.run(train_once.summary_op, feed_dict=feed_dict)
          #timer2.print()
          train_once.summary_writer.add_summary(summary_str, step)
      else:
        #get eval loss for every batch eval, then add train loss for eval step average loss
        summary_str = sess.run(train_once.summary_op, feed_dict=eval_feed_dict) if train_once.summary_op is not None else ''
        #all single value results will be add to summary here not using tf.scalar_summary..
        summary.ParseFromString(summary_str)
        melt.add_summarys(summary, eval_results, eval_names_, suffix='eval')

      melt.add_summarys(summary, train_average_loss, names_, suffix='train_avg%dsteps'%eval_interval_steps) 

      if metric_evaluate:
        melt.add_summarys(summary, evaluate_results, evaluate_names, prefix='evaluate')
      
      train_once.summary_writer.add_summary(summary, step)
      train_once.summary_writer.flush()

      #timer_.print()
    
    if print_time:
      full_duration = train_once.eval_timer.elapsed()
      if metric_evaluate:
        metric_full_duration = train_once.metric_eval_timer.elapsed()
      full_duration_str = 'elapsed:{:.3f} '.format(full_duration)
      #info.write('duration:{:.3f} '.format(timer.elapsed()))
      duration = timer.elapsed()
      info.write('duration:{:.3f} '.format(duration))
      info.write(full_duration_str)
      info.write('eval_time_ratio:{:.3f} '.format(duration/full_duration))
      if metric_evaluate:
        info.write('metric_time_ratio:{:.3f} '.format(duration/metric_full_duration))
    #print(gezi.now_time(), epoch_str, 'eval_step: %d'%step, info.getvalue())
    logging.info2('{} {} {}'.format(epoch_str, 'eval_step: %d'%step, info.getvalue()))

    return stop
