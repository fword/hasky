#!/bin/sh
source ./hadoop.config

input=$train_data_path

output=$train_output_path
$HADOOP_HOME/bin/hadoop fs -test -e $output
if [ $? -eq 0 ];then
	$HADOOP_HOME/bin/hadoop fs -rmr $output
fi
$HADOOP_HOME/bin/hadoop fs -mkdir $output

output=$train_count_path
$HADOOP_HOME/bin/hadoop fs -test -e $output
if [ $? -eq 0 ];then
	$HADOOP_HOME/bin/hadoop fs -rmr $output
fi

$HADOOP_HOME/bin/hadoop streaming \
	-D mapred.hce.replace.streaming="false" \
	-input $input \
	-output $output \
	-mapper "sh ./hadoop-write-bin.sh gen-records-streaming.py $train_output_path train 0" \
	-reducer "$python ./count_reducer.py" \
	-file "./hadoop.config" \
	-file "./hadoop-write-bin.sh" \
	-file "./count_reducer.py" \
	-file "./conf.py" \
	-file "./gen-records-streaming.py" \
	-file "$local_train_output_path/vocab.bin" \
	-cacheArchive "$root/lib/deepiu.tar.gz#." \
	-cacheArchive "$root/resource/data.tar.gz#." \
	-cacheArchive "$root/resource/conf.tar.gz#." \
	-jobconf stream.memory.limit=1200 \
	-jobconf mapred.job.name="chenghuige-prepare-train" \
	-jobconf mapred.reduce.tasks=100 \
	-jobconf mapred.map.tasks=1999 \
	-jobconf mapred.reduce.tasks=1 \
	-jobconf mapred.job.map.capacity=5000 \
	-jobconf mapred.job.reduce.capacity=100 \
	-jobconf mapred.map.capacity.per.tasktracker=1 \
	-jobconf mapred.reduce.capacity.per.tasktracker=1 \
	-jobconf mapred.max.map.failures.percent=10 \
	-jobconf mapred.max.reduce.failures.percent=5 

$HADOOP_HOME/bin/hadoop fs -get $train_count_path/*  $local_train_output_path/num_records.hdfs.txt

#@TODO not work, local ok, hadoop cut: you must specify a list of bytes, characters, or fields
#-reducer "bash -c 'cut -f 2 | paste -sd+ | bc'" \

#-file "$local_train_output_path/vocab.bin" \

