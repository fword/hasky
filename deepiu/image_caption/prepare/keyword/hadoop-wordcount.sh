#!/bin/sh
source ./hadoop.config

input=$train_data_path

output=$train_word_count_path

min_count=10

$HADOOP_HOME/bin/hadoop fs -test -e $output
if [ $? -eq 0 ];then
	$HADOOP_HOME/bin/hadoop fs -rmr $output
fi

$HADOOP_HOME/bin/hadoop streaming \
	-D mapred.hce.replace.streaming="false" \
	-input $input \
	-output $output \
	-mapper "$python ./wordcount_mapper.py --seg_method $seg_method" \
	-reducer "$python ./wordcount_reducer.py $min_count" \
	-file "./hadoop.config" \
	-file "./wordcount_mapper.py" \
	-file "wordcount_reducer.py" \
	-file "./conf.py" \
	-cacheArchive "$root/lib/deepiu.tar.gz#." \
	-cacheArchive "$root/resource/data.tar.gz#." \
	-cacheArchive "$root/resource/conf.tar.gz#." \
	-jobconf stream.memory.limit=1200 \
	-jobconf mapred.job.name="chenghuige-wordcount" \
	-jobconf mapred.job.priority=VERY_HIGH \
	-jobconf mapred.reduce.tasks=100 \
	-jobconf mapred.map.tasks=5000 \
	-jobconf mapred.reduce.tasks=100 \
	-jobconf mapred.job.map.capacity=2000 \
	-jobconf mapred.job.reduce.capacity=1000 \
	-jobconf mapred.reduce.capacity.per.tasktracker=1 \
	-jobconf mapred.max.map.failures.percent=5 \
	-jobconf mapred.max.reduce.failures.percent=5 

$HADOOP_HOME/bin/hadoop fs -getmerge $train_word_count_path  $local_train_output_path/vocab.hdfs.ori.txt

#@TODO not work, local ok, hadoop cut: you must specify a list of bytes, characters, or fields
#-reducer "bash -c 'cut -f 2 | paste -sd+ | bc'" \
#-file "/home/gezi/temp/image-caption/keyword/bow/train.v0/vocab.bin" \

#-jobconf mapred.map.capacity.per.tasktracker=1 \
#-jobconf mapred.reduce.capacity.per.tasktracker=1 \
#-reducer "$python ./wordcount_reducer.py $min_count" \
#-jobconf mapred.max.reduce.failures.percent=5 
#partitioner com.baidu.sos.mapred.lib.IntHashPartitioner \
