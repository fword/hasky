cp ./prepare/flickr/conf.py conf.py
source ./prepare/flickr/config 
model_dir=~/data/models/flickr.rnn2/ 
mkdir -p $model_dir
cp ./train-flickr-rnn-good.sh $model_dir

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
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 10 \
	--keep_interval 1 \
  --num_negs 1 \
  --use_neg 0 \
  --debug 0 \
  --feed_dict 0 \
  --algo rnn \
  --interval 100 \
  --eval_interval 500 \
  --margin 0.5 \
  --learning_rate 0.002 \
  --seg_method en \
  --feed_single 0 \
  --dynamic_batch_length 1 \
	--batch_size 16 \
  --rnn_method 0 \
  --monitor_level 2 \
  --add_global_scope 0 \


  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
