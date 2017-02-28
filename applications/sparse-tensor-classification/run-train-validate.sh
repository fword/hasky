#pushd .
#cd ../tf-record/sparse/ 
#sh run.sh 
#popd 

#--for store tensorboard info
mkdir -p /tmp/model 

python ./train-validate.py ../../data/text-classification/tf-record/train  ../../data/text-classification/tf-record/test 
python ./train-validate-shared.py ../../data/text-classification/tf-record/train  ../../data/text-classification/tf-record/test 

