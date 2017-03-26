from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
class DataSet(object):
  def __init__(self, samples):
    print("samples.shape: " , samples.shape)
    self._num_samples = samples.shape[0]
    self._Spin_number = samples.shape[1]
    self._samples = samples
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def samples(self):
    return self._samples
  @property
  def num_samples(self):
    return self._num_samples
  @property
  def Spin_number(self):
    return self._Spin_number


  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_samples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_samples)
      numpy.random.shuffle(perm)
      self._samples = self._samples[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_samples
    end = self._index_in_epoch
    return self._samples[start:end]

def read_data_sets(train_dir,filename):
  class DataSets(object):
    pass
  data_sets = DataSets()
  TRAIN_set = numpy.load(train_dir+filename)
  data_sets.train = DataSet(TRAIN_set)
  return data_sets