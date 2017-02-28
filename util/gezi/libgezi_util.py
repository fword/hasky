#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   libgezi_util.py
#        \author   chenghuige  
#          \date   2016-08-25 21:08:23.051951
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi.nowarning

"""
@TODO gezi should remove boost.python dependence
now has libgezi_util and segment  depend on boost.python
"""

#@FIXME this might casue double free at the end, conflict with numpy in virutal env

import libstring_util as su 

def get_single_cns(text):
  return su.to_cnvec(su.extract_chinese(text)) 

