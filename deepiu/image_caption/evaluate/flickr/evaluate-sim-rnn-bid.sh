python ./evaluate-sim.py \
  --image_url_prefix 'D:\data\image-text-sim\flickr\imgs\' \
  --print_predict 0 \
  --algo rnn \
  --model_dir $1 \
  --topn $2 \
  --add_global_scope 1 \
  --batch_size 20 \
  --num_examples $3 \
  --rnn_method 2 \
  
