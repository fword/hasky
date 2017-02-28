import subprocess, time, sys

command = 'sh ./train/cuda1.sh sh ./train/hdfs-keyword-rnn.phrase.50w.sh'

check_interval = 10

class Superviser():
  def __init__(self):
    self.run()
    while True:
      time.sleep(check_interval)
      if self.p.poll():
        self.run()
  def run(self):
    print('start running!', command)
    self.p = subprocess.Popen(command, shell=True)

worker = Superviser()
