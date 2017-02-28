tensorflow/pytorch related work(nlp and image related, text classification, image caption, seq2seq..)   

## incase not find dependence, make sure set PYTHONPATH to include tensorflow_exp/util so we can find gezi and melt
##./examples/ 
./examples/tf-record/  
show how to write and read TFRecord(tensorflow standard dataa format)   
./examples/sparse-tensor-classification/  
this is self contained mlp classification example showing   
how to read sparse TFRecord and train a mlp classifier 
./examples/text-classification  
reading libsvm format then do text classification  

Notice some examples like text-classification will move to deepiu

##./deepiu
main application root  
###./deepiu/image-caption
image-caption related work now support    
discriminant method:  
bow,  
rnn,  
cnn(TODO)  
generative method:  
show_and_tell  
show_attend_and_tell(TODO)  
Input with both image and text(TODO) 
###./deepiu/text-sum
app with long text as input(like image title, ct0) and predict shortter summary text(like click query)  
supporting method:  
seq2seq
seq2seq_attetion   
seq2seq_attetion_copy(TODO)  
###./deepiu/seq2seq 
common seq2seq codes used for image-caption, text-sum and other applications
###./deepiu/util
common util for deepiu  

##./util
###./util/gezi
common lib 
###./util/melt
common tensorflow related lib, you can view it similar as tf.contrib
####./util/melt/flow
like  tf.supervisor, make train and test flow easier,  
main functions include, application only make graph and pass train_ops and evaluate_ops to flow  
flow will do model save, log save(auto add all one dimentional shape tensors to tensorboard), show elapsed time ...  

## publish lib
1. gezi ./util/gezi 
2. melt ./util/melt
3. hasky ./util/hasky  
4. deepiu ./deepiu  
