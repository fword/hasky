source ./config 

python ./inference/inference-outgraph.py \
      --decode_max_words 10 \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
