cp ../../prepare/keyword/app-conf/seq-basic.10w/conf.py conf.py 
source ../../prepare/keyword/app-conf/seq-basic.10w/config

python ./evaluate-sim.py \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --valid_resource_dir $valid_output_path \
  --vocab=/home/gezi/work/keywords/train/v2/zhongce/models/showandtell/vocab.bin \
  --print_predict=0 \
  --algo=show_and_tell \
  --model_dir=/home/gezi/work/keywords/train/v2/zhongce/models/showandtell \
  --batch_size 1 \
  
