source ./config 

python ./inference/inference.py \
      --model_dir '/home/gezi/temp/textsum/model.seq2seq.attention' \
      --debug 1 \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
