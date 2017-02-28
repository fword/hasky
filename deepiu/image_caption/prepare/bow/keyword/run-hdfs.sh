sh ./gen-vocab-hdfs.sh 
sh ./hadoop-prepare-train-countonly.sh &
sh ./hadoop-prepare-train-binonly.sh &
#sh ./hadoop-prepare-train.sh &
sh ./run-local.sh
