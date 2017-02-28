#!/usr/bin/env python
# ==============================================================================
#          \file   __init__.py
#        \author   chenghuige  
#          \date   2016-08-16 16:36:38.289129
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from melt.layers.layers import *

from melt.layers.optimizers_backward_compat import *

#TODO
#if int(tf.__version__.split('.')[1]) > 10:
#  from melt.layers.optimizers import *
#else:
#  from melt.layers.optimizers_backward_compat import *
