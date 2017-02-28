cp  ../../prepare/seq-with-unk/flickr/for_keyword_conf.py conf.py
source ../../prepare/seq-with-unk/flickr/evalkeyword.config 
#python ./evaluate-generation-for-keyword.py --batch_size 1000 --image_feature_file=/home/gezi/data/image-caption/keyword/manyidu-2016q2/img2fea.txt --image_url_prefix=D:\\data\\image-caption\\keyword\\manyidu-2016q2\\imgs\\
python ./evaluate-generation-for-keyword.py \
  --batch_size 1000 \
  --image_feature_file=/home/gezi/data/image-caption/keyword/evaluate-big/img2fea.txt \
  --image_url_prefix=D:\\data\\image-caption\\keyword\\evaluate-big\\imgs\\
