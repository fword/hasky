source ./config 

mkdir -p $fixed_valid_output_path

cp ./conf.py ../../


fixed_valid_input=$input_path/evaluate/test

cp ./conf.py ../../../
python ./gen-records.py \
  --threads 1 \
  --input $fixed_valid_input \
  --vocab $train_output_path/vocab.txt \
  --output $fixed_valid_output_path \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --name test
