source ./config 

mkdir -p $train_output_path

python ./gen-records.py  \
  --input $train_data_path/train \
  --vocab $train_output_path/vocab.bin \
  --output $train_output_path \
  --name train \
  #--save_np 0

python ./gen-distinct-texts.py --dir $train_output_path 
