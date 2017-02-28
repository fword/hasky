source ./config 
mkdir -p $train_output_path
cat $train_data_path/train_* | python ./gen-vocab.py --out_dir $train_output_path --most_common $vocab_size --seg_method $seg_method
