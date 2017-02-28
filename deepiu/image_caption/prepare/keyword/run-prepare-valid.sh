source ./config 
ln -s $resource_dir/data .
ln -s $resource_dir/conf .

cp ./conf.py ../../../
sh ./prepare-valid.sh 

rm data 
rm conf 
