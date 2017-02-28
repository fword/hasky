data_path=~/data/text-classification/
out_path=$data_path/tf-record 
python ./gen-records.py $data_path/train.10w $out_path/train
python ./gen-records.py $data_path/test.1w $out_path/test 
python ./read-records-melt.py $out_path/test --num_epochs=2
