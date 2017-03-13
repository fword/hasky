#train with attention   
sh ./train/seq2seq-attention.sh  
#inference with attention  
## inference output generated text/seq   
sh ./inference/inference-attention.sh    
## inference output score of in_seq -> out_seq prob  
sh ./inference/inference-score-attention.sh  
