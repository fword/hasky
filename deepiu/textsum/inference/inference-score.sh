source ./config 

python ./inference/inference-score.py \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
