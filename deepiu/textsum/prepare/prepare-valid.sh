source ./config 
mkdir -p $valid_output_path

cp ./conf.py ../../

python ./gen-records.py  \
  --input=$valid_data_path/'test' \
  --vocab=$train_output_path/'vocab.txt' \
  --output=$valid_output_path \
  --max_lines=100000 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --name test

python ./gen-distinct-texts.py --dir=$valid_output_path --shuffle $shuffle_texts --max_texts $max_texts

