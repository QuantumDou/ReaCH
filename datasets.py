import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from test_utils import clean_train_report,clean_iu_test_report,clean_mimic_test_report

class BaseDataset(Dataset):
    def __init__(self,args, tokenizer, split):  
        self.tokenizer = tokenizer
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_len = args.max_seq_len
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.examples)


class IuxrayTrainDataset(BaseDataset):
    def __getitem__(self, i):
        sample = self.examples[i]
        text_a = sample['question'].lower() 
        answer = sample['answer']
        report = clean_train_report(sample['report'])  
        image_path = sample['image_path']
        q_segment_id = self.tokenizer.get_id_by_token('<question>')
        e_segment_id = self.tokenizer.get_id_by_token('<explanation>')

        
        tokens = self.tokenizer.tokenize(text_a)  
        labels = [-100] * len(tokens)  
        segment_ids = [q_segment_id] * len(tokens)
        ques_mask = [1] * len(tokens)
        ques_end = len(tokens)
        
        answer_token = ['<bos>'] + self.tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer_token)
        tokens += answer_token
        ans_end = len(tokens)

        report_token = self.tokenizer.tokenize(" so the report is " + report) + ['<eos>']
        report_len = len(report_token)
        tokens += report_token     
        cap_end = len(tokens)-1

        labels += [-100] + answer_token[1:] + report_token  
        segment_ids += [e_segment_id] * answer_len + report_len * [e_segment_id]   
        ques_mask += [0] * answer_len + report_len * [0]

        assert len(tokens) == len(segment_ids)
        assert len(tokens) == len(labels)

       
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]
            ques_mask = ques_mask[:self.max_seq_len]
        if cap_end >= self.max_seq_len:
            cap_end = self.max_seq_len -1

       
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len   
        tokens = tokens + (['<pad>'] * padding_len)
        labels = labels + ([-100] * padding_len) 
        segment_ids += ([e_segment_id] * padding_len)  
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        ques_mask += [0] * padding_len   
        ques_mask = torch.tensor(ques_mask, dtype=torch.long)

        dec_mask = [0] * ques_end + [1] * (seq_len - ques_end) + [0] * padding_len  
        dec_mask = torch.tensor(dec_mask, dtype=torch.long)


        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.get_id_by_token(t) if t != -100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
 
        answer_idx = torch.zeros(len(self.tokenizer.idx2token))
        answer_idx[labels[ques_end+4]] = 1  

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)

        return (image, input_ids, labels, segment_ids, ques_end, ans_end,cap_end, dec_mask,ques_mask,answer_idx)



class IuxrayTestDataset(BaseDataset):
    def __getitem__(self, i):
        sample = self.examples[i]
        text_a = sample['question'].lower()  
        text_b = clean_iu_test_report(sample['report'])  
        image_path = sample['image_path']
        q_segment_id = self.tokenizer.get_id_by_token('<question>')
        e_segment_id = self.tokenizer.get_id_by_token('<explanation>')

        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        prompt = ['<bos>'] + self.tokenizer.tokenize(" the answer is ")
        prompt_len = len(prompt)
        tokens += prompt

        segment_ids += [e_segment_id] * prompt_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        report_ids = self.tokenizer.tokenize(text_b) + ['<eos>'] 
        report_ids = self.tokenizer.convert_tokens_to_ids(report_ids)[:60]
        report_ids = torch.tensor(report_ids, dtype=torch.long)
       
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        
        return (image, input_ids, report_ids, segment_ids)





class MimicTrainDataset(BaseDataset):
    def __getitem__(self, i):
        sample = self.examples[i]
        text_a = sample['question'].lower() 
        answer = sample['answer']
        report = clean_train_report(sample['report'])  
        image_path = sample['image_path']
        q_segment_id = self.tokenizer.get_id_by_token('<question>')
        e_segment_id = self.tokenizer.get_id_by_token('<explanation>')

        
        tokens = self.tokenizer.tokenize(text_a)   
        labels = [-100] * len(tokens) 
        segment_ids = [q_segment_id] * len(tokens)
        ques_mask = [1] * len(tokens)
        ques_end = len(tokens)
        
        answer_token = ['<bos>'] + self.tokenizer.tokenize(" the answer is " + answer)
        answer_len = len(answer_token)
        tokens += answer_token
        ans_end = len(tokens)

        report_token = self.tokenizer.tokenize(" so the report is " + report) + ['<eos>']
        report_len = len(report_token)
        tokens += report_token    
        cap_end = len(tokens)-1

        labels += [-100] + answer_token[1:] + report_token  
        segment_ids += [e_segment_id] * answer_len + report_len * [e_segment_id]   
        ques_mask += [0] * answer_len + report_len * [0]

        assert len(tokens) == len(segment_ids)
        assert len(tokens) == len(labels)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]
            ques_mask = ques_mask[:self.max_seq_len]
        if cap_end >= self.max_seq_len:
            cap_end = self.max_seq_len -1

        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len   
        tokens = tokens + (['<pad>'] * padding_len) 
        labels = labels + ([-100] * padding_len) 
        segment_ids += ([e_segment_id] * padding_len) 
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        ques_mask += [0] * padding_len  
        ques_mask = torch.tensor(ques_mask, dtype=torch.long)

      
        dec_mask = [0] * ques_end + [1] * (seq_len - ques_end) + [0] * padding_len  
        dec_mask = torch.tensor(dec_mask, dtype=torch.long)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.get_id_by_token(t) if t != -100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)

       
        answer_idx = torch.zeros(len(self.tokenizer.idx2token))
        answer_idx[labels[ques_end+4]] = 1

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return (image, input_ids, labels, segment_ids, ques_end, ans_end,cap_end,dec_mask,ques_mask,answer_idx)

class MimicTestDataset(BaseDataset):
    def __getitem__(self, i):
        sample = self.examples[i]
        text_a = sample['question'].lower() 
        text_b = clean_mimic_test_report(sample['report']) 
        image_path = sample['image_path']
        q_segment_id = self.tokenizer.get_id_by_token('<question>')
        e_segment_id = self.tokenizer.get_id_by_token('<explanation>')

        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        prompt = ['<bos>']
        prompt_len = len(prompt)
        tokens += prompt

        segment_ids += [e_segment_id] * prompt_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        report_ids = self.tokenizer.tokenize(text_b) + ['<eos>'] 
        report_ids = self.tokenizer.convert_tokens_to_ids(report_ids)[:60]
        report_ids = torch.tensor(report_ids, dtype=torch.long)
    
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, input_ids, report_ids, segment_ids)
