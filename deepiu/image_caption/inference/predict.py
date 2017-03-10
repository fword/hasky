#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-06 19:47:14.278205
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
#need resource of Segmentor or will check fail.. here not use segmentor for en
#need vocab file
#need model 

#input file of images
#input file of texts

#python predict.py text_file model_dir vocab
#sys.stdin will be image feature file

import numpy as np
from libword_counter import Vocabulary 
import gezi

from deepiu.image_caption.algos import algos_factory


WORDS_SEP = ' '
TEXT_MAX_WORDS = 80
NUM_RESERVED_IDS = 1 
ENCODE_UNK = 0
IMAGE_FEATURE_LEN = 1000

vocabulary = Vocabulary(sys.argv[3], NUM_RESERVED_IDS)

algo = 'bow'
predictor = algos_factory.gen_predictor(algo)
predictor.init_predict(TEXT_MAX_WORDS)
predictor.load(sys.argv[2])

ids_list = []
for line in open(sys.argv[1]):
  line = line.strip().split('\t')[-1]
  words = line.split()
  ids = [vocabulary.id(word) for word in text.split(WORDS_SEP) if vocabulary.has(word) or ENCODE_UNK]
  ids = gezi.pad(ids, TEXT_MAX_WORDS)
  ids_list.append(ids)

ids_list = np.array(ids_list)

for line in sys.stdin:
  l = line.strip().split('\t')
  image_names = l[0]
  image_features = np.array([[float(x) for x in l[1:]]])



