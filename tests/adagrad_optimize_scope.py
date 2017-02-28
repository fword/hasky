import tensorflow as tf
import numpy as np

def build_graph(x_data, y_data):
  # Try to find values for W and b that compute y_data = W * x_data + b
  # (We know that W should be 0.1 and b 0.3, but TensorFlow will
  # figure that out for us.)
  W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
  b = tf.Variable(tf.zeros([1]))
  y = W * x_data + b

  # Minimize the mean squared errors.
  loss = tf.reduce_mean(tf.square(y - y_data))

  return loss 

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


loss = build_graph(x_data, y_data)

tf.get_variable_scope().reuse_variables()

eval_loss = build_graph(x_data, y_data)

train = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=None,
        learning_rate=tf.constant(0.1),
        optimizer=tf.train.AdagradOptimizer)  

#Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
  _, loss_, eval_loss_ = sess.run([train, loss, eval_loss])
  if step % 20 == 0:
    print(step, loss_, eval_loss_)

# Learns best fit is W: [0.1], b: [0.3]