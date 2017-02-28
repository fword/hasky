#conf_dir=$1
#cp  $conf_dir/* .

source ./config

sh ./ln.sh

sh ./gen-vocab-hdfs.sh 

#sh ./hadoop-prepare-valid-countonly.sh
#sh ./hadoop-prepare-valid-binonly.sh
sh ./hadoop-prepare-train-countonly.sh &
sh ./hadoop-prepare-train-binonly.sh &

#sh ./hadoop-prepare-train.sh &

sh ./run-local-valid.sh

