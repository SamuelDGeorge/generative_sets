
import tensorflow as tf
import numpy as np


def get_file_lists(data_dir):
    import glob
 
    train_list = glob.glob(data_dir + '/' + 'Training_Set-*')
    valid_list = glob.glob(data_dir + '/' + 'Validation_Set-*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list
 
def _parse_record(raw_record, is_training, code_len, set_len):
    """Parse a set from `value`."""
    keys_to_features = {
      'set/code': tf.FixedLenFeature([code_len * set_len], dtype=tf.float32, default_value=np.zeros(code_len * set_len)),
      'set/file': tf.FixedLenFeature((), tf.string, default_value='Dog.jpeg,Dog.jpeg,Dog.jpeg,'),
      'set/classes':tf.FixedLenFeature([set_len], dtype=tf.int64, default_value=np.zeros(set_len)),
      'set/num_unique': tf.FixedLenFeature([1], dtype=tf.int64, default_value=np.zeros(1)),
      'set/code_size': tf.FixedLenFeature([1], dtype=tf.int64, default_value=np.zeros(1)),
      'set/set_size': tf.FixedLenFeature([1], dtype=tf.int64, default_value=np.zeros(1))
    }
    
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    code_size = parsed['set/code_size']
    set_size = parsed['set/set_size']
    
    code = tf.cast(
        tf.reshape(parsed['set/code'], shape=[set_len, code_len]),
        dtype=tf.float16) 
    
    classes = tf.cast(
        tf.reshape(parsed['set/classes'], shape=[set_len]),
        dtype=tf.int16)
    
    #make file into 
    files = parsed['set/file']
    #files = files.split(',')[0:set_size] 
    
    uniques = tf.cast(
        tf.reshape(parsed['set/num_unique'], shape=[]),
        dtype=tf.int32)
    
    
    
    return code, classes, files, uniques 

def get_set_batch(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    dataset = tf.data.TFRecordDataset(filenames)
 
    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)
 
    dataset = dataset.map(lambda value: parse_record(value, is_training),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
 
    features, labels = iterator.get_next()
 
    return features, labels

def build_set_iterator(is_training, filenames, batch_size, num_epochs=1000, num_parallel_calls=12):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: _parse_record(value, is_training),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator

def build_set_dataset(is_training, filenames, code_size=2048, set_size=3,batch_size=30, num_epochs=1000, num_parallel_calls=12):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: _parse_record(value, is_training, code_size, set_size),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset

