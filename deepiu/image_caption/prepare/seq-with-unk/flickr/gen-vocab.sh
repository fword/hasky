source ./config 
mkdir -p $train_output_path
python ./gen-vocab.py  $train_data_path/results_20130124.token --out_dir $train_output_path --min_count 30
