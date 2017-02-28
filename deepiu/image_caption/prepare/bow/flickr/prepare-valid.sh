source ./config 

#rm -rf $valid_output_path
mkdir -p $valid_output_path


python ./gen-records.py \
  --image $valid_data_path/img2fea.txt \
  --text $valid_data_path/results_20130124.token \
  --ori_text_index -1 \
  --vocab $train_output_path/vocab.bin \
  --out $valid_output_path \
  --name test

python ./gen-distinct-texts.py --dir $valid_output_path 
