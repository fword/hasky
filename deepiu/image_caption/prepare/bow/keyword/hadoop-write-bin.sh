#!/bin/sh
source ./hadoop.config

$python ./$1 --output_directory ./output --part $mapred_task_partition --name $3 --mode $4

output=$2 

if [ $4 -ne 2 ];then
	$HADOOP_HOME/bin/hadoop fs -test -e $output/*_$mapred_task_partition
	if [ $? -ne 0 ];then
		$HADOOP_HOME/bin/hadoop dfs -Dhadoop.job.ugi=$ugi -put ./output/* $output
	fi
fi

