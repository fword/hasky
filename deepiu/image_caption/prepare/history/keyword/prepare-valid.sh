source ./config 
mkdir -p $valid_output_path

python ./gen-records.py  \
  --input $valid_data_path/test \
  --vocab $train_output_path/vocab.bin \
  --output $valid_output_path \
  --name test

python ./gen-distinct-texts.py --dir $valid_output_path 
