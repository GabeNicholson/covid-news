import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer)
from covid import RobertaClassifier, SentimentData
tqdm.pandas()

def tokenizer_mp(sent, tokenizer):
    return len(tokenizer.encode(sent, add_special_tokens=True))

def tokenize_covid_dataframe(dataframe, headlines=False):
    dataframe.page_num.fillna('None', inplace=True)
    print((dataframe.isna().sum() == 0).all() == True)
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Instantiate Finetuned Classifier
    finetuned_rob = RobertaClassifier().to(device)
    finetuned_rob.load_state_dict(torch.load('/home/ec2-user/SageMaker/pre_trained_model/covid_checkpoint.pth', map_location=device));
    tokenizer = RobertaTokenizer.from_pretrained("/home/ec2-user/SageMaker/pre_trained_tokenizer")
        
    if headlines:
        with mp.Pool(mp.cpu_count()) as pool:
            prod_x = partial(tokenizer_mp, tokenizer=tokenizer)
            dataframe['len_tokenized'] = pool.map(prod_x, dataframe['title'])    
        dataframe = dataframe[dataframe['len_tokenized'] < 500]
        dataframe['sentences'] = dataframe['title']
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            prod_x = partial(tokenizer_mp, tokenizer=tokenizer)
            dataframe['len_tokenized'] = pool.map(prod_x, dataframe['pairs'])
        dataframe = dataframe[(dataframe['len_tokenized'] < 500) & (dataframe['len_tokenized'] > 20)]
        dataframe['sentences'] = dataframe['pairs']
    
    MAX_LEN = dataframe.len_tokenized.max()
    print(f'Max length of tokenized pair sentences: {MAX_LEN}')
    print(f'Percentage of sentences with a tokenized length greater than 300 {len(dataframe[dataframe.len_tokenized > 300])/len(dataframe)}.')
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe, device, tokenizer, finetuned_rob
    
    
def regular_news_tokenize(dataframe):
    dataframe.dropna(inplace=True)
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Instantiate Finetuned Classifier
    tokenizer = RobertaTokenizer.from_pretrained("/home/ec2-user/SageMaker/non-covid-model/tokenizer");
    model = RobertaForSequenceClassification.from_pretrained('/home/ec2-user/SageMaker/non-covid-model').to(device);

    with mp.Pool(mp.cpu_count()) as pool:
        prod_x = partial(tokenizer_mp, tokenizer=tokenizer)
        dataframe['len_tokenized'] = pool.map(prod_x, dataframe['pairs'])    
        
    dataframe = dataframe[(dataframe['len_tokenized'] < 500) & (dataframe['len_tokenized'] > 20)]
    MAX_LEN = dataframe.len_tokenized.max()
    subset_df = dataframe # Using entire timeline.
    subset_df.reset_index(drop=True,inplace=True)
    
    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts
        
        def __len__(self):
            return len(self.tokenized_texts["input_ids"])
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}

    pred_texts = subset_df.pairs.astype('str').tolist()
    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True, max_length=MAX_LEN)
    pred_dataset = SimpleDataset(tokenized_texts)
    trainer = Trainer(model=model)
    predictions = trainer.predict(pred_dataset)

    preds = predictions.predictions.argmax(-1)
    # labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    t_df = pd.DataFrame(list(zip(preds,scores)), columns=['prediction','score'])
    predicted_df = pd.concat((subset_df,t_df), axis=1)
    predicted_df = predicted_df[['date', 'article_id', 'title', 'prediction', 'score']]
    return predicted_df
    