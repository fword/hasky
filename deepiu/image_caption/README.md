@TODO should modify comment prepare
re segment in gen-records.py as keyword

take flickr dataset for example
1. got to ~/data/image-text-sim/flickr/,
   sh run.sh to genearte idl 4w feature(for each image output 1000 dim feature which is cnn classification final ouput before softmax)
   sh split-train-test.sh  
2. go to ./prepare/flickr/ 
   sh run.sh 
   will make all train ,valid, fix_valid data ok
3. ./read-records.py  to test if step 2 genearted data is as expected
4. sh train/train-flickr-show-and-tell.sh 


@FIXME
you can reproduce this strange fail of loss.. 
./train-flickr-rnn2-debug.sh
using /home/gezi/temp/image-caption/model.flickr.rnn2.nan


python supervise.py 'cuda0.sh ./train/hdfs-keyword-bow.sh'
