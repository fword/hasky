cp ./prepare/seq-with-unk/flickr/conf.py conf.py
source ./prepare/seq-with-unk/flickr/config 

#model_dir=./model.flickr.show_and_tell2
model_dir=./model.flickr

python ./test.py \
  --train_input=$valid_output_path/'test*' \
  --vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix 'D:\data\image-text-sim\flickr\imgs\' \
  --model_dir $model_dir \
  --train_only 1 \
  --show_eval 0 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 3 \
  --num_evaluate_examples 10 \
  --keep_interval 1 \
  --num_negs 1 \
  --use_neg 0 \
  --per_example_loss 1 \
  --debug 0 \
  --num_epochs 1 \
  --feed_dict 0 \
  --algo show_and_tell \
  --interval 100 \
  --eval_interval 200 \
  --margin 0.5 \
  --optimizer adagrad \
  --learning_rate 0.01 \
  --seg_method en \
  --feed_single 0 \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --batch_size 100 \
  --num_sampled 10000 \
  --log_uniform_sample 1 \
  --monitor_level 2 \

