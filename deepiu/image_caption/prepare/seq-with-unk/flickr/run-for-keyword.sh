cp ./for_keyword_conf.py conf.py 
cp ./evalkeyword.config config 

source ./evalkeyword.config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

echo 'From train data dir:', $train_data_path
echo 'Will write to train output dir:', $train_output_path

mkdir -p $dir

sh ./gen-vocab.sh 

sh ./prepare-fixed-valid.sh
sh ./prepare-valid.sh 
sh ./prepare-train.sh 

rm data 
rm conf 

cp ./for_keyword_conf.py ./conf.py 
cp ./normal.config ./config 
