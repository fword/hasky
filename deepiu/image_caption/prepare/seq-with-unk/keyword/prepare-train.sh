source ./config 

mkdir -p $train_output_path

cp ./conf.py ../../../
python ./gen-records.py  \
  --input $train_data_path/train \
  --vocab $train_output_path/vocab.bin \
  --output $train_output_path \
  --name train \
  --np_save 0

#python ./gen-distinct-texts.py --dir $train_output_path 
