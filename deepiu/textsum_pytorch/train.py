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
flags.DEFINE_string('valid_input', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/valid/test_*', '')
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


def train_once_(sess, step, input_text, text, model, optimizer):
  if not hasattr(train_once_, 'train_loss'):
    train_once_.train_loss = 0.

  if not hasattr(train_once_, 'summary_writter'):
    log_dir = '/home/gezi/temp/textsum_pytorch'
    train_once_.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

  summary = tf.Summary()

  pred = model(input_text, text, feed_previous=False)
  
  total_loss = None
  time_steps = text.size()[1]
  batch_size = len(text)
  for time_step in xrange(time_steps - 1):
    y_pred = pred[time_step]
    target = text[:, time_step + 1]
    loss = criterion(y_pred, target)
    if total_loss is None:
      total_loss = loss
    else:
      total_loss += loss 
  optimizer.zero_grad()
  total_loss /= batch_size
  #print('loss', total_loss)
  total_loss.backward()
  optimizer.step()
  #NOTICE! must be .data[0] other wise will consume more and more gpu mem, see 
  #https://discuss.pytorch.org/t/cuda-memory-continuously-increases-when-net-images-called-in-every-iteration/501
  #https://discuss.pytorch.org/t/understanding-graphs-and-state/224/1
  train_once_.train_loss += total_loss.data[0]

  steps = 10
  if step % steps == 0:
    avg_loss = train_once_.train_loss if step is 0 else train_once_.train_loss / steps
    print('step:', step, 'train_loss:', avg_loss)
    train_once_.train_loss = 0.
    
    names = melt.adjust_names([avg_loss], None)
    melt.add_summarys(summary, [avg_loss], names, suffix='train_avg%dsteps'%steps) 
    train_once_.summary_writer.add_summary(summary, step)
    train_once_.summary_writer.flush()


def train_once(sess, step, ops, model, optimizer):
  input_text, text = read_once(sess, step, ops)
  train_once_(sess, step, input_text, text, model, optimizer)


from melt.flow import tf_flow
import input
def read_records():
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

  ops = inputs(
    FLAGS.input,
    decode=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_threads,
    #num_threads=1,
    batch_join=FLAGS.batch_join,
    shuffle_batch=FLAGS.shuffle_batch,
    shuffle=FLAGS.shuffle,
    #fix_random=True,
    #fix_sequence=True,
    #no_random=True,
    allow_smaller_final_batch=True,
    )
  print(ops) 
  
  timer = Timer()
  tf_flow(lambda sess, step: train_once(sess, step, ops, model, optimizer))
  print(timer.elapsed())
    

def main(_):
  read_records()

if __name__ == '__main__':
  tf.app.run()
