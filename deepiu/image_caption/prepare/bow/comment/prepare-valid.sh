source ./config 
mkdir -p $valid_output_path

python ./to_flickr_caption.py $valid_data_path/result_total_gbk.txt > $valid_output_path/results_20130124.token  --seg_method $seg_method

python ./gen-records.py \
  --image $valid_data_path/img2fea.txt \
  --text $valid_output_path/results_20130124.token \
  --vocab $train_output_path/vocab.bin \
  --out $valid_output_path \
  --name test

python ./gen-distinct-texts.py --dir $valid_output_path 

#----for fixed valid
python ./gen-records.py \
  --image $valid_data_path/img2fea.txt \
  --text $valid_output_path/results_20130124.token \
  --vocab $train_output_path/vocab.bin \
  --out $fixed_valid_output_path \
  --name test \
  --threads 1 \
  --num_records 10 \
