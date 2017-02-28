source ./config 

mkdir -p $fixed_valid_output_path

fixed_valid_input=/home/gezi/data/image-caption/keyword/evaluate/result.txt

python ./gen-records.py \
  --threads 1 \
  --input $fixed_valid_input \
  --vocab $train_output_path/vocab.bin \
  --output $fixed_valid_output_path \
  --name test
