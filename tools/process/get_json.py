import random 
import json
import csv
import ast
from get_answer import clean_disease_type,clean_normal_type
import re
from string import punctuation
import pickle

def parse_decimal(text):
    find_float = lambda x: re.search("\d+(\.\d+)", x).group()
    text_list = []
    for word in text.split():
        try:
            decimal = find_float(word)
            new_decimal = decimal.replace(".", "*")
            text_list.append(new_decimal)
        except:
            text_list.append(word)
    return " ".join(text_list)

def clean_train_sentence(text):
    punc = list(punctuation)
    text = re.sub(r"xxxx", " ", text)
    text = re.sub("[^a-z\s]", "", text.lower())
    text_nopunc = [char for char in text if char not in punc] 
    text_nopunc = "".join(text_nopunc)
    wd = []
    for word in text_nopunc.split():
        wd.append(word)
    sentence = " ".join(wd)
    if sentence.strip()=='images':
        return []
    return sentence

def clean_train_report(report):
        report = parse_decimal(report)
        report_cleaner = lambda t: t.replace('. .', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('.')
        
        tokens = [clean_train_sentence(sent) for sent in report_cleaner(report) if sent!='' if clean_train_sentence(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report
    
def select_question(i):
    return ques[i]

def filter_fields(data, fields_to_keep):
    return {k: {field: v[field] for field in fields_to_keep if field in v} for k, v in data.items()}


with open('../question/iu_xray/random_list_train.pkl', 'rb') as f:
    random_list_train = pickle.load(f)
with open('../question/iu_xray/random_list_val.pkl', 'rb') as f:
    random_list_val = pickle.load(f)
with open('../question/iu_xray/random_list_test.pkl', 'rb') as f:
    random_list_test = pickle.load(f)


tag_dict = {}
with open('../meta_data/iu_xray_major_tags.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    for key,value in data.items():
        parts = key.split('-')
        img_id = '-'.join(parts[:-1])
        if value==["normal"]:
             tag =  0  
        else :
             tag =  1   
        tag_dict[img_id] = tag
        
        
ques=[]
with open('../question/question.json', 'r') as file:
    ques_data = json.load(file)
    ques = ques_data["question"]

train_data = {}
val_data = {}
test_data = {}
train_count,val_count,test_count=0,0,0
with open("iuxray.csv", mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row['split'] == 'train':
            if row['id'] not in train_data:
                train_data[row['id']] = {
                    'id':row['id'],
                    'split': row['split'],
                    'image_path': ast.literal_eval(row['image_path']),
                    'report':clean_train_report(row['report']),
                    'question': select_question(random_list_train[train_count]),
                    'disease_pair':row['pairs'],
                    'tag':tag_dict[row['id']],
                    'answer':clean_normal_type(row['pairs']) if tag_dict[row['id']] ==0 else clean_disease_type(row['pairs'])
                }
                train_count+=1

        if row['split'] == 'val':
            if row['id'] not in val_data:
                val_data[row['id']] = {
                    'id':row['id'],
                    'split': row['split'],
                    'image_path': ast.literal_eval(row['image_path']),
                    'report':row['report'],
                    'question': select_question(random_list_val[val_count]),
                    'disease_pair':row['pairs'],
                    'tag':tag_dict[row['id']],
                    'answer':clean_normal_type(row['pairs']) if tag_dict[row['id']] ==0 else clean_disease_type(row['pairs'])
                }
                val_count+=1

        if row['split'] == 'test':
            if row['id'] not in test_data:
                test_data[row['id']] = {
                    'id':row['id'],
                    'split': row['split'],
                    'image_path': ast.literal_eval(row['image_path']),
                    'report':row['report'],
                    'question': select_question(random_list_test[test_count]),
                    'disease_pair':row['pairs'],
                    'tag':tag_dict[row['id']],
                    'answer':clean_normal_type(row['pairs']) if tag_dict[row['id']] ==0 else clean_disease_type(row['pairs'])
                }
                test_count+=1


fields_to_keep = ['id', 'image_path', 'report','tag','question','disease_pair','answer','split']
filtered_train_data = filter_fields(train_data, fields_to_keep)
filtered_val_data = filter_fields(val_data, fields_to_keep)
filtered_test_data = filter_fields(test_data, fields_to_keep)

combined_data = {
    'train': list(filtered_train_data.values()),
    'val': list(filtered_val_data.values()),
    'test': list(filtered_test_data.values())
}

with open("../../dataset/iu_xray/IU_CDRC.json", mode='w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)