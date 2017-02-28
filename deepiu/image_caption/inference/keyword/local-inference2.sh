source ./hadoop.config
cp $model/conf.py .
cp $model/config .
cp $model/vocab.bin .
source ./config 

cat ./simid_imgfeature_keywords2.txt | python ./predict.py --model_dir=$model --algo=$algo --seg_method=$online_seg_method --feed_single=$feed_single --image_feature_place=$image_feature_place --text_place=$text_place
