source ./config 
mkdir -p $valid_output_path

python ./gen-records.py  \
  --input=$valid_data_path/'test' \
  --vocab=$train_output_path/'vocab.bin' \
  --output=$valid_output_path \
  --name test

python ./gen-distinct-texts.py --dir=$valid_output_path 

python ./gen-image-labels.py  \
  --input=$valid_data_path/'test' \
  --output=$valid_output_path/'image_labels.npy'

python ./gen-image-names-and-features.py \
  --input=$valid_data_path/'test' \
  --output_dir=$valid_output_path 

python ./gen-bidirectional-label-map.py  \
  --input=$valid_data_path/'test' \
  --all_distinct_text_strs=$valid_output_path/'distinct_text_strs.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' 
 