source ./config 

mkdir -p $train_output_path

cp ./conf.py ../../
python ./gen-records.py  \
  --input $train_data_path/train \
  --vocab $train_output_path/vocab.txt \
  --output $train_output_path \
  --max_lines=0 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --name train \
  --np_save 0

#python ./gen-distinct-texts.py --dir $train_output_path 
