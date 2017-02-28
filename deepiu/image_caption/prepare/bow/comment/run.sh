source ./config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

echo 'From train data dir:', $train_data_path
echo 'Will write to train output dir:', $train_output_path

rm -rf /tmp/*.comment

sh ./prepare-train.sh 
sh ./prepare-valid.sh 

rm data 
rm conf 
