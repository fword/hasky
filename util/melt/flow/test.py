#!/usr/bin/env python
# ==============================================================================
#          \file   test_tfrecord.py
#        \author   chenghuige  
#          \date   2016-08-16 13:01:09.932544
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf 
import gezi
from gezi import AvgScore
import melt
#not used yet 
from melt.flow.flow import tf_test_flow

def test_flow(ops, names=None, gen_feed_dict=None, deal_results=None, model_dir='./model', 
              model_name=None, num_epochs=1, num_interval_steps=100, eval_times=0,
              print_avg_loss=True, sess=None):
  """
  test flow, @TODO improve list result print
  Args:
  ops: eval ops
  names: eval names
  model_path: can be dir like ./model will fetch lates model in model dir , or be real model path like ./model/model.0.ckpt
  @TODO num_epochs should be 1,but now has problem of loading model if set @FIXME, so now 0
  """
  if sess is None:
    sess = tf.InteractiveSession()
  melt.restore(sess, model_dir, model_name)

  if not os.path.isdir(model_dir):
    model_dir = os.path.dirname(model_dir)
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(model_dir, sess.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  try:
    step = 0
    eval_step = 0
    avg_eval = AvgScore()
    total_avg_eval = AvgScore()
    while not coord.should_stop():
      feed_dict = {} if gen_feed_dict is None else gen_feed_dict()
      results = sess.run(ops, feed_dict=feed_dict)
      if not isinstance(results, (list, tuple)):
        results = [results]
      
      if deal_results is not None:
        #@TODO may need to pass summary_writer, and step
        #use **args ?
      	  deal_results(results)
      
      if print_avg_loss:
        results = gezi.get_singles(results)
        avg_eval.add(results)
        total_avg_eval.add(results)
        if step % num_interval_steps == 0:
          average_eval = avg_eval.avg_score()
          print('{}: average evals = {}'.format(gezi.now_time(), melt.value_name_list_str(average_eval, names)), 'step:', step)
          summary = tf.Summary()
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary.ParseFromString(summary_str)
          for i in xrange(len(results)):
            name = i if names is None else names[i]
            summary.value.add(tag='metric{}'.format(name), simple_value=average_eval[i])
          summary_writer.add_summary(summary, step)
 
          if eval_step and eval_step == eval_times:
            break
          eval_step += 1
      step += 1
    print('Done testing for {} epochs, {} steps. AverageEvals:{}'.format(num_epochs, step, gezi.pretty_floats(total_avg_eval.avg_score())))
  except tf.errors.OutOfRangeError:
    print('Done testing for {} epochs, {} steps. AverageEvals:{}'.format(num_epochs, step, gezi.pretty_floats(total_avg_eval.avg_score())))
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  # Wait for threads to finish.
  coord.join(threads)
