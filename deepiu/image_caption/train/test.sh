cp ./prepare/flickr/conf.py conf.py
source ./prepare/flickr/config 

model_dir=/home/gezi/data/models/model.flickr.bow/

python ./test.py \
	--train_input $valid_output_path/'test*' \
	--valid_input '' \
	--vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix 'D:\data\image-text-sim\flickr\imgs\' \
  --model_dir $model_dir \
	--show_eval 0 \
	--batch_size 100 \
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
  --num_interval_steps 100 \
  --eval_times 0 \

