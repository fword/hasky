cp ./prepare/bow/flickr/conf.py conf.py
source ./prepare/bow/flickr/config 

dir=/home/gezi/temp/image-caption/ 
model_dir=$dir/model.flickr.rnn
cp ./train-flickr-rnn.sh $model_dir

python ./test.py \
	--train_input=$valid_output_path/'test*' \
	--vocab=$train_output_path/vocab.bin \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --model_dir=$model_dir \
  --num_epocs 1 \
  --show_eval 0 \
  --train_only 1 \
  --num_negs 1 \
  --use_neg 0 \
  --debug 1 \
  --feed_dict 0 \
  --algo rnn \
  --interval 100 \
  --eval_interval 1000 \
  --margin 0.5 \
  --learning_rate 0.01 \
  --seg_method en \
  --feed_single 0 \
  --dynamic_batch_length 1 \
  --batch_size 16 \
  --rnn_method 0 \
  --monitor_level 2 \

