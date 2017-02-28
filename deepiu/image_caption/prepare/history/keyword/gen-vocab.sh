source ./config 
mkdir -p $train_output_path
python ./gen-vocab.py  $train_data_path/train --out_dir $train_output_path --most_common $vocab_size
