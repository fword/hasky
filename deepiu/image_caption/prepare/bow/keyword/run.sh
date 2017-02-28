source ./config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

echo 'From train data dir:', $train_data_path
echo 'Will write to train output dir:', $train_output_path

#sh ./gen-vocab.sh 

sh ./prepare-fixed-valid.sh
sh ./prepare-valid.sh
sh ./prepare-train.sh 

rm data 
rm conf 
