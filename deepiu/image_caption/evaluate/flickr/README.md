how to compare two checkpoint with differnt steps but same model ?

python ./reset-model-top-scope.py --model_dir ./model.flickr.show_and_tell2/model.ckpt-8000 --scope show_and_tell --index 1
python ./evaluate-generation-compare.py --m1 ../../model.flickr.show_and_tell2/ --m2 /tmp/model.ckpt-8000 --index 1

so /tmp/model.ckpt-8000 in show_and_tell_1 and neweast model in show_and_tell

~/mine/tensorflow-exp/models/image-caption/evaluate/flickr$ python ./evaluate-sim.py   --image_url_prefix='D:\data\image-text-sim\flickr\imgs\'   --valid_resource_dir $valid_output_path   --vocab=$train_output_path/vocab.bin   --print_predict=0   --algo=rnn  --model_dir=/home/gezi/temp.local/image-caption/model.flickr.rsum  --rnn_output_method 0 --rnn_method 1 

state of the art...
['ndcg@1:0.120', 'ndcg@5:0.101', 'ndcg@10:0.132', 'avg_precision:0.115']
