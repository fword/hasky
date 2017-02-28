source ./config 

mkdir -p $train_output_path

python ./gen-records.py \
	--image $train_data_path/img2fea.txt \
	--text $train_data_path/results_20130124.token \
	--vocab $train_output_path/vocab.bin \
  --ori_text_index -1 \
	--out $train_output_path \
	--name train 

python ./gen-distinct-texts.py --dir $train_output_path 
