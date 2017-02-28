data_path=~/data/E2006
out_path=$data_path/tf-record 
python ./gen-records.py $data_path/E2006.train/E2006.train $out_path/train --label_type=float
python ./gen-records.py $data_path/E2006.test/E2006.test $out_path/test --label_type=float
python ./read-records-melt.py $out_path/test --num_epochs=2 --label_type=float
