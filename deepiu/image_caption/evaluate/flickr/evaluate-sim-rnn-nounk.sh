cp ../../prepare/bow/flickr/conf.py conf.py
source ../../prepare/bow/flickr/config 

python ./evaluate-sim.py \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --valid_resource_dir $valid_output_path \
  --vocab=$train_output_path/vocab.bin \
  --print_predict=0 \
  --algo=rnn \
  --model_dir=$1 \
  --batch_size 1000 \
  
