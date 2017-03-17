
# coding: utf-8

# In[1]:

import tensorflow as tf
sess = tf.InteractiveSession()


# In[2]:

values = tf.constant([[[ 2.44353771,  0.65329778, -2.08763194, -0.64653987,  0.50398952,
        -0.66223699,  2.24809909, -0.60570312, -0.74135935,  2.5471828 ,
         2.91521454, -1.30316925,  0.81219608],
       [-0.06042745, -1.43421483,  1.98799515,  0.46523282,  1.17595863,
         0.43213287, -0.33148676,  0.36731029, -2.01692128,  0.75370973,
         2.03620768, -0.97226971, -0.90286547]]])


# In[3]:

values.eval()


# In[4]:

values.eval().shape


# In[5]:

zeros = tf.zeros_like(values, dtype=tf.float32)


# In[6]:

values2 = tf.concat([values, zeros], 1)


# In[7]:

values2.eval()


# In[8]:

values2.eval().shape


# In[9]:

import melt


# In[10]:

with tf.variable_scope('abc') as scope:
    keys = tf.contrib.layers.linear(
            values, 13, biases_initializer=None)
    print keys
    #scope.reuse_variables()
    keys2 = tf.contrib.layers.linear(
        values2, 13, biases_initializer=None)
    print keys2


# In[11]:

sess.run(tf.global_variables_initializer())


# In[12]:

keys.eval()


# In[13]:

keys.eval().shape


# In[14]:

keys2.eval()


# In[15]:

for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print item


# In[ ]:



