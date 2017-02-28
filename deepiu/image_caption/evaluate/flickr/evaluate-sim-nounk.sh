cp ../../prepare/bow/flickr/conf.py conf.py
source ../../prepare/bow/flickr/config 

python ./evaluate-sim-metagraph.py \
  --print_predict=0 \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --valid_resource_dir=$valid_output_path \
  --vocab=$train_output_path/vocab.bin \
  --model_dir=$1 \
  --algo=$2 \
  
