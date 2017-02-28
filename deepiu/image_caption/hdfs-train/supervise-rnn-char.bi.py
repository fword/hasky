import subprocess, time, sys

command = './train/cuda2.sh ./hdfs-train/keyword-rnn-char.bi.sh'

check_interval = 10

class Superviser():
  def __init__(self):
    self.run()
    while True:
      time.sleep(check_interval)
      if self.p.poll():
        self.run()
  def run(self):
    print('start running!')
    self.p = subprocess.Popen(command, shell=True)

worker = Superviser()
