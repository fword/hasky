cp ./prepare/seq-with-unk/flickr/conf.py conf.py
source ./prepare/seq-with-unk/flickr/config 

dir=/home/gezi/temp/image-caption/ 
model_dir=$dir/model.flickr.show_and_tell
mkdir -p $model_dir
cp ./train-flickr-showandtell.sh $model_dir

python ./train.py \
	--train_input=$train_output_path/'train*' \
	--valid_input=$valid_output_path/'test*' \
	--fixed_valid_input=$fixed_valid_output_path/'test*' \
	--valid_resource_dir=$valid_output_path \
	--vocab=$train_output_path/vocab.bin \
  --num_records_file=$train_output_path/num_records.txt \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --model_dir $model_dir \
  --show_eval 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 10 \
  --keep_interval 0.5 \
  --monitor_level 2 \
  --num_negs 1 \
  --use_neg 0 \
  --debug 0 \
  --feed_dict 0 \
  --algo show_and_tell \
  --interval 100 \
  --eval_interval 1000\
  --seg_method en \
  --feed_single 0 \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --margin 0.5 \
  --learning_rate 0.01 \
  --batch_size 16 \
  --per_example_loss 1 \
  --num_layers 1 \
  --keep_prob 1 \
  --num_sampled 0 \ 
  --log_uniform_sample 1 \

