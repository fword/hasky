source ./config 

mkdir -p $fixed_valid_output_path

python ./gen-records.py \
  --threads 1 \
  --input ~/data/image-text-sim/evaluate/result.txt \
  --vocab $train_output_path/vocab.bin \
  --output $fixed_valid_output_path \
  --name test
