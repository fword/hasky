sh ./gen-vocab-hdfs.sh 
#sh ./hadoop-prepare-valid-countonly.sh
#sh ./hadoop-prepare-valid-binonly.sh
sh ./hadoop-prepare-train-countonly.sh &
sh ./hadoop-prepare-train-binonly.sh &
sh ./run-local.sh
