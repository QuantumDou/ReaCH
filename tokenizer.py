import json
import re
from collections import Counter
import torch
from test_utils import clean_train_report

class Tokenizer(object):
    def __init__(self, args):
        
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()



    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann['train']:
            
          
            str = example['question'].lower() + ' the answer is '+ example['answer'] + ' so the report is '+ clean_train_report(example['report'])
           
            tokens = str.split()
            
            for token in tokens:
                total_tokens.append(token)
       
        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx 
            idx2token[idx] = token
        return token2idx, idx2token

        
    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    # tokenizer(report)   str->[ids]
    def __call__(self, report):  
        tokens = report.split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = ids 
        return ids
    
    # str->[tokens]
    def tokenize(self,report):
        tokens = report.split()
        return tokens


    # [tokens]->[ids]
    def convert_tokens_to_ids(self,tokens):
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = ids 
        return ids

    def decode(self, ids,skip_special_tokens=False):
        SPECIAL_TOKENS = ['<unk>', '<bos>', '<eos>', '<pad>','<question>', '<answer>', '<explanation>']
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        txt = ''
        for i, idx in enumerate(ids):
            
            if i >= 1 and self.idx2token[idx] not in SPECIAL_TOKENS:
                txt += ' '  
            new_txt = self.idx2token[idx]
            if not skip_special_tokens:
                txt += new_txt
            else:
                if new_txt in SPECIAL_TOKENS:
                    continue
                else:
                    txt += new_txt

           
        return txt

    def decode_batch(self, ids_batch,skip_special_tokens=False):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids,skip_special_tokens))
        return out
