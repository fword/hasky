import tensorflow as tf 
def decode_examples(batch_serialized_examples):
  features = tf.parse_example(
    batch_serialized_examples,
    features={
        'label' : tf.FixedLenFeature([], tf.int64),
        'feature' : tf.FixedLenFeature([488], tf.float32),
    })

  label = features['label']
  feature = features['feature']

  return feature, label
