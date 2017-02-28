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

flags.DEFINE_string('input', '/tmp/train/train_*', '')
flags.DEFINE_string('output', '/tmp/train/num_records.txt', '')
flags.DEFINE_integer('threads', 12, '')

import sys, os, time
import gezi
import melt 

import  multiprocessing
from multiprocessing import Process, Manager, Value

counter = Value('i', 0)
def deal_file(file):
  count = melt.get_num_records_single(file)
  global counter
  with counter.get_lock():
    counter.value += count 
  print(file, count)

def main(_):
  timer = gezi.Timer()
  input = sys.argv[1] if len(sys.argv) == 2 else FLAGS.input 
  
  if FLAGS.threads == 1:
    num_records = melt.get_num_records_print(input)
    print(timer.elapsed())
  else:
    files = gezi.bigdata_util.list_files(input)
    print(files)
    pool = multiprocessing.Pool(processes = FLAGS.threads)
    pool.map(deal_file, files)
    pool.close()
    pool.join()
    
    num_records = counter.value 
    print('num_records:', num_records)

  output = FLAGS.output if gezi.bigdata_util.is_remote_path(input) \
      else os.path.join(os.path.dirname(input), 'num_records.txt')
  print('write to %s'%output)
  out = open(output, 'w')
  out.write(str(num_records))


if __name__ == '__main__':
  tf.app.run()
