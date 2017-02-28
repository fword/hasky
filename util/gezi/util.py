#!/usr/bin/env python
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-18 18:24:05.771671
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys, os
import numpy as np

def add_one(d, word):
  if not word in d:
    d[word] = 1
  else:
    d[word] += 1

def pretty_floats(values):
  if not isinstance(values, (list, tuple)):
    values = [values]
  return [float('{:.3f}'.format(x)) for x in values]
  #return ['{}'.format(x) for x in values]

def get_singles(l):
  """
  get signle elment as list, filter list
  """
  return [x for x in l if not isinstance(x, collections.Iterable)]

def is_single(item):
  return not isinstance(item, collections.Iterable)

def iterable(item):
  """
  be careful!  like string 'abc' is iterable! 
  you may need to use if not isinstance(values, (list, tuple)):
  """
  return isinstance(item, collections.Iterable)

def is_list_or_tuple(item):
  return isinstance(item, (list, tuple))

def get_value_name_list(values, names):
  return ['{}:{:.3f}'.format(x[0], x[1]) for x in zip(names, values)]

#@TODO better pad 
def pad(l, maxlen, mark=0):
  if isinstance(l, list):
    l.extend([mark] * (maxlen - len(l)))
    return l[:maxlen]
  elif isinstance(l, np.ndarray):
    return l
  else:
    raise ValueError('not support')

def nppad(l, maxlen):
  if maxlen > len(l):
    return np.lib.pad(l, (0, maxlen - len(l)), 'constant')
  else:
    return l[:maxlen]

def try_mkdir(dir):
  if not os.path.exists(dir):
    print('make new dir: [%s]'%dir, file=sys.stderr)
    os.makedirs(dir)

def get_dir(path):
  if os.path.isfile(path):
    return os.path.dirname(path)
  else:
    return path

#@TODO perf?
def dedupe_list(l):
  #l_set = list(set(l))
  #l_set.sort(key = l.index)
  l_set = []
  set_  = set()
  for item in l:
    if item not in set_:
      set_.add(item)
      l_set.append(item)
  return l_set

#@TODO
def parallel_run(target, args_list, num_threads):
  record = []
  for thread_index in xrange(num_threads):
    process = multiprocessing.Process(target=target,args=args_list[thread_index])
    process.start()
    record.append(process)
  
  for process in record:
    process.join()

import threading
def multithreads_run(target, args_list):
  num_threads = len(args_list)
  threads = []
  for args in args_list:
    t = threading.Thread(target=target, args=args) 
    threads.append(t) 
  for t in threads:
    t.join()

#@TODO move to bigdata_util.py


#----------------file related
def is_glob_pattern(input):
  return '*' in input

import os
import glob
def list_files(input):
  if os.path.isdir(input):
    return os.listdir(input)
  elif os.path.isfile(input):
    return [input]
  else:
    return glob.glob(input)

def sorted_ls(path, time_descending=True):
  mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
  return list(sorted(os.listdir(path), key=mtime, reverse=time_descending))

def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  files = [file for file in glob.glob('%s/model.ckpt-*'%(model_dir)) if not file.endswith('.meta')]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files 
  
#----------conf
def save_conf(con):
  file = '%s.py'%con.__name__
  out = open(file, 'w')
  for key,value in con.__dict__.items():
    if not key.startswith('__'):
      if not isinstance(value, str):
        result = '{} = {}\n'.format(key, value)
      else:
        result = '{} = \'{}\'\n'.format(key, value)
      out.write(result)

def write_to_txt(data, file):
  out = open(file, 'w')
  out.write('{}'.format(data))

def read_int_from(file):
  return int(open(file).readline().strip().split()[0]) if os.path.isfile(file) else 0

def read_float_from(file):
  return float(open(file).readline().strip().split()[0]) if os.path.isfile(file) else 0

def read_str_from(file):
  return open(file).readline().strip() if os.path.isfile(file) else 0

def img_html(img):
  return '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n'.format(img)

def text_html(text):
  return '<p>{}</p>'.format(text)

def thtml(text):
  return text_html(text)

#@TODO support *content 
def hprint(content):
  print('<p>', content,'</p>')

def imgprint(img):
  print(img_html(img))


def unison_shuffle(a, b):
  """
  http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
  """
  assert len(a) == len(b)
  try:
    from sklearn.utils import shuffle
    a, b = shuffle(a, b, random_state=0)
    return a, b
  except Exception:
    print('sklearn not installed! use numpy but is not inplace shuffle', file=sys.stderr)
    import numpy
    index = numpy.random.permutation(len(a))
    return a[index], b[index]

def finalize_feature(fe, mode='w', outfile='./feature_name.txt', sep='\n'):
  #print(fe.Str('\n'), file=sys.stderr)
  #print('\n'.join(['{}:{}'.format(i, fname) for i, fname in enumerate(fe.names())]), file=sys.stderr)
  #print(fe.Length(), file=sys.stderr)
  if mode == 'w':
    fe.write_names(file=outfile, sep=sep)
  elif mode == 'a':
    fe.append_names(file=outfile, sep=sep)

def write_feature_names(names, mode='a', outfile='./feature_name.txt', sep='\n'):
  out = open(outfile, mode)
  out.write(sep.join(names))
  out.write('\n')

def get_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names

def read_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names

def get_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict

def read_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict

def update_sparse_feature(feature, num_pre_features):
  features = feature.split(',')
  index_values = [x.split(':') for x in features]
  return ','.join(['{}:{}'.format(int(index) + num_pre_features, value) for index,value in index_values])

def merge_sparse_feature(fe1, fe2, num_fe1):
  if not fe1:
    return update_sparse_feature(fe2, num_fe1)
  if not fe2:
    return fe1 
  return ','.join([fe1, update_sparse_feature(fe2, num_fe1)])


#TODO move to other place
#http://blog.csdn.net/luo123n/article/details/9999481
def edit_distance(first,second):  
  if len(first) > len(second):  
    first,second = second,first  
  if len(first) == 0:  
    return len(second)  
  if len(second) == 0:  
    return len(first)  
  first_length = len(first) + 1  
  second_length = len(second) + 1  
  distance_matrix = [range(second_length) for x in range(first_length)]   
  for i in range(1,first_length):  
    for j in range(1,second_length):  
      deletion = distance_matrix[i-1][j] + 1  
      insertion = distance_matrix[i][j-1] + 1  
      substitution = distance_matrix[i-1][j-1]  
      if first[i-1] != second[j-1]:  
        substitution += 1  
      distance_matrix[i][j] = min(insertion,deletion,substitution)  
  return distance_matrix[first_length-1][second_length-1]  
