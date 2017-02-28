from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

import sys
print('tensorflow_version:', tf.__version__, file=sys.stderr)

import melt.utils
from melt.utils import logging

from melt.util import *
from melt.ops import *
from melt.variable import * 
from melt.tfrecords import * 

from melt.inference import *

import melt.layers

import melt.slim

import melt.flow
from melt.flow import projector_config

import melt.metrics 
from melt.metrics import *

import melt.apps

import melt.rnn 

import melt.seq2seq
