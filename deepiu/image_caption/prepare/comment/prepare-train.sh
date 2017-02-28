source ./config 

echo 'From train data dir:', $train_data_path
echo 'Will write to train output dir:', $train_output_path
mkdir -p $train_output_path

python ./to_flickr_caption.py $train_data_path/result_total_gbk.txt > $train_output_path/results_20130124.token  --seg_method $seg_method
python ./gen-vocab.py  $train_output_path/results_20130124.token --out_dir $train_output_path --most_common 1000 

python ./gen-records.py \
	--image $train_data_path/img2fea.txt \
	--text $train_output_path/results_20130124.token \
	--vocab $train_output_path/vocab.bin \
	--out $train_output_path \
	--name train 

python ./gen-distinct-texts.py --dir $train_output_path 

