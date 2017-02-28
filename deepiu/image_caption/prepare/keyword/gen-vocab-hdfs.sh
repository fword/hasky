source ./config 
sh ./hadoop-wordcount.sh
mkdir -p $train_output_path
cat $train_output_path/vocab.hdfs.ori.txt | python ./gen-vocab-hdfs.py --out_dir $train_output_path --most_common $vocab_size 
mv $train_output_path/vocab.hdfs.bin $train_output_path/vocab.bin
mv $train_output_path/vocab.hdfs.txt $train_output_path/vocab.txt
