rm -rf ./models.tar.gz
tar czvf models.tar.gz models
hadoop fs -rmr /app/tuku/chenghuige/resource/models.tar.gz
hadoop fs -put models.tar.gz /app/tuku/chenghuige/resource
