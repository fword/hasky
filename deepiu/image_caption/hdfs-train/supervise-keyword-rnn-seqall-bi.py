import subprocess, time, sys

command = 'CUDA_VISIBLE_DEVICES=2 sh  ./hdfs-train/keyword-rnn-seqall-bi.sh'
#command = 'sh  ./hdfs-train/keyword-rnn-seqall-bi.sh'

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
