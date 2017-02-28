cp ./prepare/flickr/conf.py conf.py
source ./prepare/flickr/config 

model_dir=~/data/models/model.flickr.bow/
mkdir -p $model_dir
cp ./train-flickr-bow.sh $model_dir

python ./train.py \
	--train_input $train_output_path/'train*' \
	--valid_input $valid_output_path/'test*' \
	--fixed_valid_input $fixed_valid_output_path/'test*' \
	--valid_resource_dir $valid_output_path \
	--vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix 'D:\data\image-text-sim\flickr\imgs\' \
  --model_dir $model_dir \
	--show_eval 1 \
	--batch_size 16 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 10 \
	--keep_interval 1 \
  --num_negs 5 \
  --use_neg 1 \
  --debug 0 \
  --algo bow \
  --interval 100 \
  --eval_interval 500 \
  --margin 0.5 \
  --feed_dict 0 \
  --seg_method en \
  --feed_single 0 \
  --combiner mean \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \
  --learning_rate 0.01 \

	#--valid_resource_dir $valid_output_path 
  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
