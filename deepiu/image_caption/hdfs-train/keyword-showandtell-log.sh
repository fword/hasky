cp ./prepare/keyword/app-conf/seq-basic.10w/conf.py conf.py
source ./prepare/keyword/app-conf/seq-basic.10w/config

dir=/home/gezi/temp/image-caption/ 
model_dir=$dir/model.keyword.showandtell.seq-basic.10w.log
mkdir -p $model_dir

python ./train.py \
	--train_input=$hdfs_train_output_path \
	--valid_input=$valid_output_path/'test_*' \
	--fixed_valid_input=$fixed_valid_output_path/'test' \
	--valid_resource_dir=$valid_output_path \
	--vocab=$train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.hdfs.txt \
  --image_url_prefix 'D:\data\image-text-sim\evaluate\imgs\' \
  --model_dir=$model_dir \
  --algo show_and_tell \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 10 \
  --num_evaluate_examples 10 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 0 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 256 \
  --num_gpus 0 \
  --min_after_dequeue 500 \
  --learning_rate 0.1 \
  --eval_interval_steps 5000 \
  --metric_eval_interval_steps 5000 \
  --save_interval_steps 5000 \
  --num_metric_eval_examples 1000 \
  --metric_eval_batch_size 1 \
  --margin 0.5 \
  --num_negs 1 \
  --use_neg 0 \
  --feed_dict 0 \
  --seg_method $seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --num_records 0 \
  --min_records 9000 \
  --log_device 0 \
  --work_mode full \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
