python ./dump-text-feature.py \
  --algo bow \
  --combiner mean \
  --model_dir ~/data/models/model.flickr.bow/ \
  --data_dir /tmp/image-caption/flickr/valid/ \
  --vocab /tmp/image-caption/flickr/train/vocab.bin \
