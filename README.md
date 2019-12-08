# Multihop Question Answering

Instructions for running the code on google colab:

The google colab notebook can be directly found here: 
[hotpot](https://colab.research.google.com/drive/1PmtSveZ0M-EzWNNTkcFh31s7B4NVh2Di).

1. Install the necessary packages
 ```
  pip install ujson
  pip install spacy
  pip install torchtext
 ```
  
2. Clone this git repository
```
git clone https://github.com/ksuma2109/CSC791.git
```
3. Copy files to the home folder
```
cp CSC791/* .
ls
```
4. Download glove embeddings. This will take around 15 minutes as the file is huge
```
chmod +x download.sh
sh download.sh
```

5. Pretrain the models for Parts of Speech tagging, Named Entity Recognition, Dependency Parsing and Semantic Entailment. **This script takes a very long time. Feel free to skip this step as the repository already contains pretrained models**. 
```
sh pretrain.sh
```

6. Preprocess train data of Hotpot Dataset
```
python main.py --mode prepro --data_file train_small.json --para_limit 2250 --data_split train
```

7. Preprocess development data of Hotpot Dataset
```
python main.py --mode prepro --data_file dev_small.json --para_limit 2250 --data_split dev
```

8. Train the model
```
python main.py --mode train --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0
```

9. Get predictions for dev dataset
```
python main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 --keep_prob 1.0 --sp_lambda 1.0 --prediction_file dev_distractor_pred_trial.json
```

10. Evaluate dev dataset
```
python hotpot_evaluate_v1.py dev_distractor_pred_trial.json hotpot_dev_distractor_v1.json
```

