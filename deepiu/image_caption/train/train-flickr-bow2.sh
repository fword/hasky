cp ./prepare/seq-with-unk/flickr/conf.py conf.py
source ./prepare/seq-with-unk/flickr/config 

dir=/home/gezi/temp/image-caption/ 
model_dir=$dir/model.flickr.bow2 
mkdir -p $model_dir
cp ./train-flickr-bow2.sh $model_dir

python ./train.py \
	--train_input=$train_output_path/'train*' \
	--valid_input=$valid_output_path/'test*' \
	--fixed_valid_input=$fixed_valid_output_path/'test*' \
	--valid_resource_dir=$valid_output_path \
	--vocab=$train_output_path/vocab.bin \
  --num_records_file=$train_output_path/num_records.txt \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --show_eval 1 \
  --batch_size 16 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 10 \
  --keep_interval 0.5 \
  --num_negs 1 \
  --use_neg 1 \
  --debug 0 \
  --algo bow \
  --interval 100 \
  --eval_interval 1000 \
  --margin 0.5 \
  --feed_dict 0 \
  --seg_method en \
  --feed_single 0 \
  --combiner=sum \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \
  --model_dir=$model_dir \
