* ./tf-record/ write and read tfrecord example 
* ./text-classification/ sparse input classification example   
* ./text-regression/  sparse input regression example
* ./classification/  dense classification example 
* ./sparse-tensor-classification/ sparse classification old, contains code without melt dependence 


Sparse clasfication training   
1. gen tfrecord   
cd ./tf-record/sparse 
sh run.sh #you may need to modify input data dir
2. train and validate  
goto ./text-classification/    
sh run.sh 
#Notice, you may also need to modify tfrecord data dir
#my dataset is 34 classes, you may need to modify num_classes and num_features in run.sh according to your own dataset


Sparse regression training
1. gen tfrecord
cd ./tf-record/sparse 
sh run.sh
#Notice we use E2006-tfidf datast for regression example which has 150360 features 
#https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html
#https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2
#https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2

2. train and validate
cd ./text-regression
sh run.sh
