https://github.com/tensorflow/tensorflow/issues/3942

I think if you have multiple files like this, tf.app.flags must be used with 'tf.app.run()'.

tf.app.run() is where the flags you have declared are parsed, so if run code that inspects FLAGS before tf.app.run(), it won't work.

My suggestion would be to set it up so that:

  model.py and util.py define functions but don't inspect FLAGS on import
  And then your 'train.py' would define its code in a main function

  def main(argv):
      .. your code ...
      Then at the bottom:

        if __name__ == "__main__":
            tf.app.run()
            As for the default values, I'll have to look
