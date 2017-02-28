#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_predictor.py
#        \author   chenghuige  
#          \date   2016-09-22 22:12:01.829820
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
 
from algos.rnn import Rnn
from algos.bow_predictor import BowPredictor

class RnnPredictor(Rnn, BowPredictor, melt.PredictorBase):