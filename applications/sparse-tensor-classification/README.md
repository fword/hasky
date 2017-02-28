## main component, notice feed.py mainly for comparaion purpose and should show same result as no feed version
* ./train.py   train with tf standard workflow of read from tf record
* ./train-feed.py  train as above but use feed dict to feed the tf record reading data to compute graph @TODO
* ./train-validate.py train and validate at same time   
* ./train-validate-shared.py train and validate at same time using get_variable
  show usage fo shared variable(instead of using class like self.w_h to share variable w_h)  
* ./train-validate-feed.py  train and validate use feed dict  @TODO 
* melt ended .py means using melt lib

## knonwn issue
* for train.py seems using cost = tf.reduce_sum will convergent much faster than using reduce_mean 
** batch 64, reduce_mean 
 python train.py ../../data/text-classification/tf-record/train --mean_cost
   step: 0 train precision@1: 0.078125 cost: 3.52427 duration: 1.03637290001
   step: 100 train precision@1: 0.21875 cost: 2.78363 duration: 0.773315906525
   step: 200 train precision@1: 0.34375 cost: 2.51744 duration: 0.766198158264
   step: 300 train precision@1: 0.421875 cost: 1.77821 duration: 0.754693984985
** batch 64, reduce_sum 
python train.py ../../data/text-classification/tf-record/train 
   step: 0 train precision@1: 0.015625 cost: 225.306 duration: 1.02878212929
   step: 100 train precision@1: 0.6875 cost: 75.5576 duration: 0.779250860214
   step: 200 train precision@1: 0.515625 cost: 82.7435 duration: 0.773854017258
   step: 300 train precision@1: 0.65625 cost: 65.3152 duration: 0.766964197159
   step: 400 train precision@1: 0.734375 cost: 57.8638 duration: 0.771269798279
   step: 500 train precision@1: 0.625 cost: 78.0657 duration: 0.808291196823
** batch 16, reduce_mean 
python train.py ../../data/text-classification/tf-record/train --mean_cost --batch_size 16
   step: 0 train precision@1: 0.0 cost: 3.52701 duration: 1.00829696655
   step: 100 train precision@1: 0.0625 cost: 3.14888 duration: 0.22234916687
   step: 200 train precision@1: 0.4375 cost: 2.33394 duration: 0.199513912201
   step: 300 train precision@1: 0.25 cost: 2.4181 duration: 0.196828126907
   step: 400 train precision@1: 0.5625 cost: 1.40977 duration: 0.195178985596
** batch 16, reduce_sum
 python train.py ../../data/text-classification/tf-record/train --batch_size 16
   step: 0 train precision@1: 0.0 cost: 56.5317 duration: 0.999857902527
   step: 100 train precision@1: 0.5625 cost: 24.9745 duration: 0.225685119629
   step: 200 train precision@1: 0.6875 cost: 16.411 duration: 0.204148054123
   step: 300 train precision@1: 0.5 cost: 23.4395 duration: 0.199006795883
   step: 400 train precision@1: 0.4375 cost: 23.4672 duration: 0.196656227112
   step: 500 train precision@1: 0.5 cost: 20.705 duration: 0.196500778198 

## @TODO
* seems reading data from tf record slower than orgnaize all your data in mem, 
  can it speed up?  
  like for batch_size 64, 
  all in mem self orgnaized gen batch and train will use about 0.3s,   
  while here will be about 0.75+  
* set num epochs not none will cause problem for model save then load, give case show this
* w_h not in cpu, will not work for gpu&adamgrad, seems sparse op only work for gpu&grad optimizer 
* dense pipline, FixedLenFeature for like 10 int? seems cifar10 example use byte_feature   
  only one string then decode_raw
