source ./config 

python ./inference/inference-score.py \
      --model_dir /home/gezi/temp/textsum/model.seq2seq.attention/ \
      --input_text_name 'seq2seq/model_init_1/input_text:0' \
      --text_name 'seq2seq/main/text:0' \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
