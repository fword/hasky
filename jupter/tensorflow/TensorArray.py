
# coding: utf-8

# In[27]:

import tensorflow as tf
print tf.__version__
sess = tf.InteractiveSession()


# In[26]:

from tensorflow.python.ops import tensor_array_ops, control_flow_ops


# In[13]:

from tensorflow.python.framework import dtypes
ta = tensor_array_ops.TensorArray(
      dtype=dtypes.float32,
      tensor_array_name="foo",
      size=3,
      infer_shape=False)

w0 = ta.write(0, [[4.0, 5.0]])
w1 = w0.write(1, [[1.0]])
w2 = w1.write(2, -3.0)

r0 = w2.read(0)
r1 = w2.read(1)
r2 = w2.read(2)


# In[8]:

r0.eval()


# In[16]:

import numpy as np
tf_dtype = dtypes.float32
dtype = tf_dtype.as_numpy_dtype()
ta = tensor_array_ops.TensorArray(
    dtype=tf_dtype, tensor_array_name="foo", size=3)

if tf_dtype == dtypes.string:
  # In Python3, np.str is unicode, while we always want bytes
  convert = lambda x: np.asarray(x).astype("|S")
else:
  convert = lambda x: np.asarray(x).astype(dtype)

w0 = ta.write(0, convert([[4.0, 5.0]]))
w1 = w0.write(1, convert([[6.0, 7.0]]))
w2 = w1.write(2, convert([[8.0, 9.0]]))


#c0 = w2.stack()
c0 = w2.pack()


# In[17]:

c0.eval()


# In[18]:

c0.eval().shape


# In[19]:

ta = tensor_array_ops.TensorArray(
    dtype=tf_dtype, tensor_array_name="foo", size=3)
w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]]))
w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
w2 = w1.write(2, convert([[8.0, 9.0]]))

c0 = w2.concat()


# In[ ]:



