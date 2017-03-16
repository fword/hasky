source ./config 

python ./inference/predict.py \
  --algo seq2seq \
  --model_dir /home/gezi/temp/textsum/model.seq2seq/ \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --rnn_method 0 \
  --cell lstm_block \
  --emb_dim 1024 \
  --rnn_hidden_size 1024 \
  --beam_size 10 \
  --decode_max_words 10 \
  --add_text_start 1 \
  --rnn_output_method 3 \
  --main_scope run \
  --use_attention 0 \
  --algo seq2seq
