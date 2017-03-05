#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os, time
from collections import defaultdict

import numpy as np

from gezi import Timer
import melt 
from deepiu.util import vocabulary 
from deepiu.textsum_pytorch import seq2seq

#flags.DEFINE_integer('batch_size', 100, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
#flags.DEFINE_integer('num_threads', 12, '')
#flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')
#flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_string('input', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/train_*', '')
flags.DEFINE_string('name', 'train', 'records name')
#flags.DEFINE_boolean('dynamic_batch_length', True, '')
#flags.DEFINE_boolean('shuffle_then_decode', True, '')

criterion = nn.NLLLoss()

vocab_size = None

def read_once(sess, step, ops):
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()

  image_name, text, text_str, input_text, input_text_str = sess.run(ops)

  input_text, text = Variable(torch.LongTensor(input_text)), Variable(torch.LongTensor(text))
 
  if torch.cuda.is_available():
    input_text, text = input_text.cuda(), text.cuda()
  
  return input_text, text


def train_once(sess, step, input_text, text, model, optimizer):
  if not hasattr(train_once, 'train_loss'):
    train_once.train_loss = 0.

  if not hasattr(train_once, 'summary_writter'):
    log_dir = '/home/gezi/temp/textsum_pytorch'
    train_once.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

  summary = tf.Summary()

  pred = model(input_text, text, feed_previous=False)
  
  total_loss = 0.
  time_steps = text.size()[1]
  batch_size = len(text)
  for time_step in xrange(time_steps - 1):
    y_pred = pred[time_step]
    target = text[:, time_step + 1]
    loss = criterion(y_pred, target)
    total_loss += loss 
  optimizer.zero_grad()
  total_loss /= batch_size
  #print('loss', total_loss)
  total_loss.backward()
  optimizer.step()
  #NOTICE! must be .data[0] other wise will consume more and more gpu mem, see 
  #https://discuss.pytorch.org/t/cuda-memory-continuously-increases-when-net-images-called-in-every-iteration/501
  #https://discuss.pytorch.org/t/understanding-graphs-and-state/224/1
  train_once.train_loss += total_loss.data[0]

  steps = FLAGS.interval_steps
  if step % steps == 0:
    avg_loss = train_once.train_loss if step is 0 else train_once.train_loss / steps
    print('step:', step, 'train_loss:', avg_loss)
    train_once.train_loss = 0.
    
    names = melt.adjust_names([avg_loss], None)
    melt.add_summarys(summary, [avg_loss], names, suffix='train_avg%dsteps'%steps) 
    train_once.summary_writer.add_summary(summary, step)
    train_once.summary_writer.flush()

def eval_once(sess, step, input_text, text, model):
  pred = model(input_text, text, feed_previous=False)
  
  time_steps = text.size()[1]
  batch_size = len(text)
  total_loss = 0.
  for time_step in xrange(time_steps - 1):
    y_pred = pred[time_step]
    target = text[:, time_step + 1]
    loss = criterion(y_pred, target) 
    total_loss += loss

  total_loss /= batch_size 
  print('step:', step, 'eval_loss', total_loss.data[0])

def process_once(sess, step, ops, eval_ops, model, optimizer):
  input_text, text = read_once(sess, step, ops)
  train_once(sess, step, input_text, text, model, optimizer)

  if eval_ops is not None:
    if step % FLAGS.eval_interval_steps == 0:
      eval_input_text, eval_text = read_once(sess, step, eval_ops)
      eval_once(sess, step, eval_input_text, eval_text, model)

from melt.flow import tf_flow
import input
import functools

def train():
  global vocab_size
  vocabulary.init()
  vocab_size = vocabulary.get_vocab_size() 

  model = seq2seq.Seq2Seq(vocab_size, FLAGS.emb_dim, 
                          FLAGS.rnn_hidden_size, 
                          FLAGS.batch_size)

  if torch.cuda.is_available():
    model.cuda()

  init_range = 0.08
  model.init_weights(init_range)
  optimizer = optim.Adagrad(model.parameters(), lr=FLAGS.learning_rate)

  inputs, decode = input.get_decodes(FLAGS.shuffle_then_decode, FLAGS.dynamic_batch_length)
  inputs = functools.partial(inputs,   
                             decode=decode,
                             num_epochs=FLAGS.num_epochs, 
                             num_threads=FLAGS.num_threads,
                             batch_join=FLAGS.batch_join,
                             shuffle_batch=FLAGS.shuffle_batch,
                             shuffle=FLAGS.shuffle,
                             allow_smaller_final_batch=True,
                             )

  ops = inputs(FLAGS.input, batch_size=FLAGS.batch_size)
  print(ops) 

  eval_ops = None
  if FLAGS.valid_input:
    #eval_ops = inputs(FLAGS.valid_input, batch_size=FLAGS.batch_size*10)
    eval_ops = inputs(FLAGS.valid_input, batch_size=FLAGS.batch_size)
  
  timer = Timer()
  tf_flow(lambda sess, step: process_once(sess, step, ops, eval_ops, model, optimizer))
  print(timer.elapsed())
    

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
