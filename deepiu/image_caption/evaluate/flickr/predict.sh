python ./predict.py \
  --vocab '/tmp/train.comment/vocab.bin' \
  --num_image 100 \
  --model_dir $1 > $2
