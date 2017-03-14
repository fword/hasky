#!/usr/bin/env python
# ==============================================================================
#          \file   beam_search.py
#        \author   chenghuige  
#          \date   2017-03-13 15:49:32.201102
#   \Description  
# ==============================================================================

"""
 now outgraph beam search copy and modify a bit from google im2txt
 TODO read and experiment https://github.com/google/seq2seq/tree/master/seq2seq for beam search
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gezi import TopN


class BeamSearchState(object):
  """Represents a complete or partial beam search state."""

  def __init__(self, words, state, logprob, score, logprobs, metadata=None):
    """Initializes the Caption.

    Args:
      words: 
      state: Model state after generating the previous word.
      logprob: Log-probability of the BeamSearchState.
      score: Score of the BeamSearchState.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'words'.
    """
    self.words = words
    self.state = state
    self.logprob = logprob
    self.score = score
    self.logprobs = logprobs
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares BeamSearchState by score."""
    assert isinstance(other, BeamSearchState)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, BeamSearchState)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, BeamSearchState)
    return self.score == other.score


def beam_search(init_states, 
                step_func, 
                end_id, 
                max_steps=20, 
                length_normalization_factor=0.):
  """
  Runs beam search caption generation on a single input
  notice beam size is not an input params since we encode it ingraph, and here 
  get it from graph

  Args:
    init_states is a tuple of (state, ids, logprobs)
    max_steps here means max decode length ie if 3 means oly can generate: A B C
  Returns:
    A list of BeamSearchState sorted by descending score.
  """
  # Feed in the image to get the initial state.
  #TODO right now ingraph beam size must here equal to out graph beam size..

  initial_state, ids, logprobs, beam_size = init_states
  partial_beams = TopN(beam_size)
  complete_beams = TopN(beam_size)
  for id, logprob in zip(ids[0], logprobs[0]):
    #assert id != end_id
    if id != end_id:
      #first id be done_id not allowed
      beam = BeamSearchState(
        words=[id],
        state=initial_state[0],
        logprob=logprob,
        score=logprob,
        logprobs=[logprob],
        metadata=[""])
      partial_beams.push(beam)

  # Run beam search. max_steps not - 1 for we wil consider <Done> as an additional step
  for _ in range(max_steps):
    partial_beams_list = partial_beams.extract()
    partial_beams.reset()
    input_feed = np.array([c.words[-1] for c in partial_beams_list])
    state_feed = np.array([c.state for c in partial_beams_list])

    state, ids, logprobs = step_func(input_feed, state_feed)

    for i, partial_beam in enumerate(partial_beams_list):
      for w, p in zip(ids[i], logprobs[i]):
        words = partial_beam.words + [w]
        logprob = partial_beam.logprob + p
        logprob_list = partial_beam.logprobs + [p]
        score = logprob
        #TODO: right now not consider metadata
        #if metadata:
        #  metadata_list = partial_caption.metadata + [metadata[i]]
        #else:
        #  metadata_list = None
        metadata_list = None
        if w == end_id:
          if length_normalization_factor > 0:
            score /= len(words)**length_normalization_factor
          beam = BeamSearchState(words, state[i], logprob, score, logprob_list, metadata_list)
          complete_beams.push(beam)
        else:
          beam = BeamSearchState(words, state[i], logprob, score, logprob_list, metadata_list)
          partial_beams.push(beam)

    #TODO: can be more greedy if not using length_normalization_factor ie if shorter path done better then
    #longger path can not be better so do not need to search
    
    #if length_normalization_factor == 0. and complete_beams.size() == beam_size:
    #  break

    if partial_beams.size() == 0:
      # We have run out of partial candidates; happens when beam_size = 1.
      break

  # If we have no complete captions then fall back to the partial captions.
  # But never output a mixture of complete and partial captions because a
  # partial caption could have a higher score than all the complete captions.
  if not complete_beams.size():
    print('----------------------warning no complete beam after max_steps:', max_steps) 
    complete_beams = partial_beams

  return complete_beams.extract(sort=True)  
