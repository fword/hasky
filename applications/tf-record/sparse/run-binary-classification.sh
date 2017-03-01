data_path=/home/gezi/data/trate
out_path=$data_path/tf-record 
mkdir -p $out_path
python ./gen-records.py $data_path/svm.tr $out_path/train --label_type=int --neg2zero=1
python ./gen-records.py $data_path/svm.te $out_path/test --label_type=int --neg2zero=1
python ./read-records-melt.py $out_path/train --num_epochs=2 --label_type=int
