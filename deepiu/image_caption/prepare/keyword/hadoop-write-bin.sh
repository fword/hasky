#!/bin/sh
source ./hadoop.config

$python ./$1 --output_directory ./output --part $mapred_task_partition --name $3 --mode $4 --seg_method=$online_seg_method --feed_single=$feed_single

output=$2 

if [ $4 -ne 2 ];then
	$HADOOP_HOME/bin/hadoop fs -test -e $output/$3_$mapred_task_partition
	if [ $? -ne 0 ];then
		$HADOOP_HOME/bin/hadoop dfs -Dhadoop.job.ugi=$ugi -put ./output/$3_$mapred_task_partition $output
	fi
fi

