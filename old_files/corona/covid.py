import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel
import torch.nn as nn
import torch.nn.functional as F

class SentimentData(Dataset):
    """Converts dataframe into usable input for pytorch

    Params:
            dataframe, tokenizer, Max_len
    
    Returns:
            'ids',
            'mask',
            'token_type_ids',
            'targets',

    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.Sentence
        # self.targets = self.data.Label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClassifier(nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        # self.l1 = RobertaModel.from_pretrained("/home/ec2-user/SageMaker/pre_trained_model")
        self.l1 = RobertaModel.from_pretrained("siebert/sentiment-roberta-large-english")
        self.pre_classifier = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(1024, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
        