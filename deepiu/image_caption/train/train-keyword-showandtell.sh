cp ./prepare/keyword/conf.py conf.py
source ./prepare/keyword/config 
python ./train.py \
	--train_input $train_output_path/'train_*' \
	--valid_input $valid_output_path/'test_*' \
	--fixed_valid_input $fixed_valid_output_path/'test' \
	--valid_resource_dir $valid_output_path \
	--vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix 'D:\data\image-text-sim\evaluate\imgs\' \
  --model_dir ./model.keyword.show_and_tell \
	--show_eval 1 \
	--batch_size 16 \
	--keep_interval 1 \
  --debug 0 \
  --algo show_and_tell \
  --interval 1000 \
  --eval_interval 10000 \
  --margin 0.5 \
  --num_negs 1 \
  --use_neg 0 \
  --seg_method phrase \
  --feed_dict 1 \
  --feed_single 1 \
  --seq_decode_method 0 \
  --dynamic_batch_length 0 \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
