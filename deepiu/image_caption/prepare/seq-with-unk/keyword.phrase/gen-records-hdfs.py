import os
import pyhdfs

name_node_address = 'nj01-nanling-hdfs.dmop.baidu.com'
name_node_port = 54310
user_name = 'tuku'
user_password = 'tuku1234'

pyhdfs.com_loadlog("./conf/", "log.conf")
fs = pyhdfs.hdfsConnectAsUser(name_node_address, name_node_port, user_name, user_password)

path = '/app/tuku/bianyunlong/clickquery/clickquery_merge_triplet.test'
path_out = '/app/tuku/chenghuige/image-text-sim/test'
hdfs_path = "hdfs://{}:{}/{}".format(name_node_address, name_node_port, path)
result, num = pyhdfs.hdfsListDirectory(fs, hdfs_path)
files = [item.mName for item in [pyhdfs.hdfsFileInfo_getitem(result, i) for i in xrange(num)]]

import glob
# for file in files:
#   print file
#   os.system('rm -rf ./test_src/*')
#   os.system('rm -rf ./test/*')  
#   os.system('hadoop fs -get %s ./test_src'%file)
#   for file_ in glob.glob('./test_src/*'):
#     print file_
#     os.system('python /home/img/chenghuige/tools/split.py %s'%file_)
#     os.system('python ./gen-records-nonpsave.py --input %s --output %s --name train'%(file_, './test'))
#   os.system('hadoop fs -put ./test/* %s'%(file_, path_out))

# for file_ in glob.glob('./test/*'):
#   if file_.endswith('.npy'):
#     continue
#   print file_
#   os.system('hadoop fs -put %s %s'%(file_, path_out))

path = '/app/tuku/bianyunlong/clickquery/clickquery_merge_triplet'
path_out = '/app/tuku/chenghuige/image-text-sim/train'
hdfs_path = "hdfs://{}:{}/{}".format(name_node_address, name_node_port, path)
result, num = pyhdfs.hdfsListDirectory(fs, hdfs_path)
files = [item.mName for item in [pyhdfs.hdfsFileInfo_getitem(result, i) for i in xrange(num)]]

import glob
index = 0
start_index = 0
for file in files:
  print file
  if index < start_index:
    index += 1
    continue
  os.system('rm -rf ./train_src/*')
  os.system('rm -rf ./train/*')  
  os.system('hadoop fs -get %s ./train_src'%file)
  for file_ in glob.glob('./train_src/*'):
    print file_
    os.system('python /home/img/chenghuige/tools/split.py %s'%file_)
    os.system('python ./gen-records-nonpsave.py --input %s --output %s --name train-%d'%(file_, './train', index))
    index += 1
    for file_ in glob.glob('./train/*'):
      print file_
      os.system('hadoop fs -put %s %s'%(file_, path_out))
    
