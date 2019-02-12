from datetime import datetime
import os
import random
import sys
import threading

import tensorflow as tf
import numpy as np

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_file_list(file_array):
    string = ''.encode('utf-8')
    for i in file_array:
        string = string + i + ','.encode('utf-8')
    return string

def convert_to_example(code, file, correct_c,num_class):
  code_size = np.array([code.shape[1]])
  set_size = np.array([code.shape[0]])
  code = code.flatten().tolist()
  file = _convert_file_list(file.flatten().tolist())
  correct_c = correct_c.flatten().tolist()
  num_class = num_class.flatten().tolist()
    
  example = tf.train.Example(features=tf.train.Features(feature={
      'set/code': _float64_feature(code),
      'set/file': _bytes_feature(tf.compat.as_bytes(file)),
      'set/classes': _int64_feature(correct_c),
      'set/num_unique': _int64_feature(num_class),
      'set/code_size': _int64_feature(code_size),
      'set/set_size': _int64_feature(set_size)}))
      
  return example



def build_set_record(name, thread_index, ranges, codes, files, correct_class, number_classes, num_shards, output_dir):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      code = codes[i]
      file = files[i]
      correct_c = correct_class[i]
      num_class = number_classes[i]

      example = convert_to_example(code, file, correct_c,
                                    num_class)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

    
def process_set_arrays(name, codes, files, correct_class, number_classes, num_shards, num_threads, output_dir):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(files) == len(codes)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(files), num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  threads = []
  for thread_index in range(len(ranges)):
    args = (name, thread_index, ranges, codes, files, correct_class, number_classes, num_shards, output_dir)
    t = threading.Thread(target=build_set_record,args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d codes in data set.' %
        (datetime.now(), len(files)))
  sys.stdout.flush()