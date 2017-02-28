#!/bin/sh
source ./hadoop.config

cp $model/vocab.bin .
cp $model/conf.py .
cp $model/config .

source ./config

$HADOOP_HOME/bin/hadoop fs -test -e $output
if [ $? -eq 0 ];then
	$HADOOP_HOME/bin/hadoop fs -rmr $output
fi

$HADOOP_HOME/bin/hadoop streaming \
	-D mapred.hce.replace.streaming="false" \
	-input $input \
	-output $output \
	-mapper "$python ./predict.py --model_dir=$model --algo=$algo --seg_method=$online_seg_method --feed_single=$feed_single --image_feature_place=$image_feature_place --text_place=$text_place" \
	-reducer NONE \
	-file "./hadoop.config" \
	-file "./predict.py" \
	-file "$model/vocab.bin" \
	-file "$model/conf.py" \
	-file "$model/config" \
	-cacheArchive "$root/lib/deepiu.tar.gz#." \
	-cacheArchive "$root/resource/data.tar.gz#." \
	-cacheArchive "$root/resource/conf.tar.gz#." \
	-cacheArchive "$root/resource/models.tar.gz#." \
	-jobconf stream.memory.limit=6000 \
	-jobconf mapred.job.name="chenghuige-inference" \
	-jobconf abaci.split.optimize.enable=true \
	-jobconf mapred.map.tasks=2000 \
	-jobconf mapred.job.map.capacity=2000 \
	-jobconf mapred.reduce.tasks=2000 \
	-jobconf mapred.job.reduce.capacity=2000 \
	-jobconf mapred.map.capacity.per.tasktracker=1 \
	-jobconf mapred.reduce.capacity.per.tasktracker=1 \
	-jobconf mapred.max.map.failures.percent=10 \
	-jobconf mapred.max.reduce.failures.percent=5 \
  -jobconf mapred.job.priority=VERY_HIGH

