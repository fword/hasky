import subprocess, time, sys

command = 'sh ./train/cuda2.sh sh ./hdfs-train/keyword-showandtell-log.sh'
#command = 'sh ./hdfs-train/keyword-showandtell-log.sh'

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
