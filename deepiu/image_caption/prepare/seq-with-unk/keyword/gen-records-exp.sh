python prepare/gen-records.py --image ./data/train/img2fea.txt --text ./data/train/results_20130124.token --vocab ./data/train/vocab.bin --out /tmp/train --name train 
python prepare/gen-records.py --image ./data/test/img2fea.txt --text ./data/test/results_20130124.token --vocab ./data/train/vocab.bin --output /tmp/test --name test 

python prepare/gen-records.py --image ./data/train/img2fea.txt --text ./data/train/results_20130124.token --vocab ./data/train/vocab.bin --out /tmp/train.sparse --name train --pad=0
python prepare/gen-records.py --image ./data/test/img2fea.txt --text ./data/test/results_20130124.token --vocab ./data/train/vocab.bin --output /tmp/test.sparse --name test --pad=0
