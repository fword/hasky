cp ./prepare/comment/conf.py conf.py
source ./prepare/comment/config 
model_dir=./model.comment.bow
cp ./train-comment-bow.sh $model_dir
python ./train.py \
	--train_input $train_output_path/'train-*' \
	--valid_input $valid_output_path/'test-*' \
	--fixed_valid_input $fixed_valid_output_path/'test-*' \
	--valid_resource_dir $valid_output_path \
	--vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix http://b.hiphotos.baidu.com/tuba/pic/item/ \
  --model_dir $model_dir \
	--fixed_eval_batch_size 10 \
	--num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 1 \
	--show_eval 1 \
	--batch_size 16 \
	--keep_interval 1 \
  --feed_dict 0 \
  --num_negs 5 \
  --debug 0 \
  --algo bow \
  --combiner sum \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \

	#--valid_resource_dir $valid_output_path \
	#--valid_resource_dir $train_output_path \
  #2> ./stderr.txt 1> ./stdout.txt
