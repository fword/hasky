source ./config 

python ./predict.py \
  --algo seq2seq \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --seg_method $seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --rnn_method 0 \
  --emb_dim 1000 \
  --rnn_hidden_size 1023 \
  --add_text_start 1 \
  --rnn_output_method 3 \
  --use_attention 1 \
  --cell lstm_block 
