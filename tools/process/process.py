import json
import re
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer

with open("process.json", "r") as f:
    js = json.load(f)
    disease_organ = js["disease_organ"]
    disease_synonym = js["disease_synonym"]
    negative_list = js["negative_list"]
    pair_synonym = js["pair_synonym"]
    organ_disease = js["organ_disease"]
    organ_synonym = js["organ_synonym"]
    positive_list = js["positive_list"]
    organs_tissue_description = js["organs_tissue_description"]
    general_disease = js["general_disease"]
# Set stopwords
# stop_words = set(stopwords.words("english"))
# Stemming
ps = PorterStemmer()
# punctuation
punc = list(punctuation)


def get_reports(report_list): 
    reports = []
    for i in range(len(report_list)):
        reports.append(report_list[i]["report"])
    return reports


def divide_to_sentences(reports):
    """
    This function is used to divide reports into several sentences.

    Args:
        reports: list[str], each str is a report

    Return:
        reports_sentences: list[list[str]], each list[str] is the divided sentences of one report
    """

    reports_sentences = []

    for report in reports:
        text_list = []

        text_new = parse_decimal(report)
        text_sentences = text_new.split(".")

        for sentence in text_sentences:
            if len(sentence) > 0:
                text_list.append(sentence)

        reports_sentences.append(text_list)

    return reports_sentences


def clean_sentence(reports):
    """
    This function is used to clean the reports.
    For example: This image doesn't show some diseases. --> This image does not show some diseases.
    """

    clean = []
    for report in reports:
        report_list = []

        for text in report:
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub("[^a-z\s]", "", text.lower())
            # Remove punctuations
            text_nopunc = [char for char in text if char not in punc] 
            text_nopunc = "".join(text_nopunc)
            wd = []
            for word in text_nopunc.split(): 
                wd.append(word)
            sentence = " ".join(wd)
            if sentence.strip()!='images':
                report_list.append(sentence) 
        clean.append(report_list) 
    return clean


def split_sentence(reports):
    """
    Split each sentence into a list of words.
    e.g.,  "a large hiatal hernia is noted" -> ['a', 'large', 'hiatal', 'hernia', 'is', 'noted', '.']
    """

    split_sen = []

    for report in reports:
        report_list = []

        for text in report:
            text_split = text.split()
            text_split.append(".")
            report_list.append(text_split)

        split_sen.append(report_list)

    return split_sen


def parse_decimal(text):
    """
    input: a sentence. e.g. "The size is 5.5 cm."
    return: a sentence. e.g. "The size is 5*5 cm."
    """

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


def pre_process(reports):
    reports_sentences = divide_to_sentences(reports)
    reports_clean = clean_sentence(reports_sentences)
    reports_split = split_sentence(reports_clean)

    return reports_split


def replace_synonym(sentence):
    sentence_list = sentence.split(" ")  
    for i in range(len(sentence_list)):
        if sentence_list[i] in disease_synonym:
            sentence_list[i] = disease_synonym[sentence_list[i]] 

    sentence = " ".join(sentence_list)

    return sentence, sentence_list


def replace_disease_synonym(sentence):
    sentence_list = sentence.split(" ")
    for i in range(len(sentence_list)):
        if sentence_list[i] in disease_synonym:
            sentence_list[i] = disease_synonym[sentence_list[i]]

    sentence = " ".join(sentence_list)
    return sentence, sentence_list


def delete_negative(sentence_organs, sentence_diseases, sentence):
    if len(sentence_diseases) == 0:
        return sentence_organs, sentence_diseases

    for negative_words in negative_list:
        if negative_words in sentence:
             
            ne_len = len(negative_words)
            app_index = sentence.find(negative_words)
            temp_sentence = sentence[ne_len + app_index:]

            temp_sentence_word_set = set(temp_sentence.split(" "))

            count = 0
            sentence_diseases_copy = sentence_diseases.copy()
            for disease in sentence_diseases_copy:
                disease_set = set(disease.split(" "))

                if disease_set.issubset(temp_sentence_word_set):
                    disease_index = sentence_diseases_copy.index(disease)

                    sentence_organs.pop(disease_index - count)

                    count += 1

    return sentence_organs, sentence_diseases


# This part is different from synonym substitution because diseases are described by more than one word


def associate_same_disease(sentence_diseases):
    # if (
    #         "enlarge" in sentence_diseases
    # ):  # enlarge a special case which is not in synonym substitution, but placed here
    #     idx = sentence_diseases.index("enlarge")
    #     sentence_diseases[idx] = "cardiomegaly"

    if "cardiomediastinal enlarged" in sentence_diseases:
        idx = sentence_diseases.index("cardiomediastinal enlarged")
        sentence_diseases[idx] = "cardiomegaly"

    if "pulmonary vascularity accentuate" in sentence_diseases:
        idx = sentence_diseases.index("pulmonary vascularity accentuate")
        sentence_diseases[idx] = "pulmonary vascularity increase"

    if "pulmonary vascularity prominent" in sentence_diseases:
        idx = sentence_diseases.index("pulmonary vascularity prominent")
        sentence_diseases[idx] = "pulmonary vascularity increase"

    return sentence_diseases


def remove_special_case(sentence_organs, sentence_diseases):
    if (
            "airspace consolidation" in sentence_diseases
            and "consolidation" in sentence_diseases
    ):
        idx = sentence_diseases.index("consolidation")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if (
            "airspace hyperinflation" in sentence_diseases
            and "hyperinflation" in sentence_diseases
    ):
        idx = sentence_diseases.index("hyperinflation")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if "airspace effusion" in sentence_diseases and "effusion" in sentence_diseases:
        idx = sentence_diseases.index("effusion")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    if (
            "granuloma calcification" in sentence_diseases
            and "granuloma" in sentence_diseases
    ):
        idx = sentence_diseases.index("granuloma")
        sentence_diseases.pop(idx)
        sentence_organs.pop(idx)

    return sentence_organs, sentence_diseases


def return_negative(sentence_organs, sentence_diseases, sentence):
    if len(sentence_diseases) == 0: 
        return sentence_organs, sentence_diseases

    for negative_words in negative_list:
        if negative_words in sentence:
            ne_len = len(negative_words)  
            app_index = sentence.find(negative_words)
            temp_sentence = sentence[ne_len + app_index:]  
            

            temp_sentence_word_set = set(temp_sentence.split(" ")) 

            count = 0
            sentence_diseases_copy = sentence_diseases.copy()
            for disease in sentence_diseases_copy:
                disease_set = set(disease.split(" "))

                if disease_set.issubset(temp_sentence_word_set):
                    disease_index = sentence_diseases_copy.index(disease)
                    negative_disease = "no " + sentence_diseases_copy[disease_index]
                    sentence_diseases[disease_index] = negative_disease

                    count += 1
    return sentence_organs, sentence_diseases


def find_diseases_organs(reports_list):  
    organs_list = []
    diseases_list = []
    reports_replaced_list = []
    reports_replaced_split_list = []

    for report in reports_list:
        report_replaced_list = []
        report_replaced_split_list = []

        for sentence in report:
            replaced_sentence, replaced_sentence_split = replace_synonym(sentence)  
            report_replaced_list.append(replaced_sentence)
            report_replaced_split_list.append(replaced_sentence_split)
            

        reports_replaced_list.append(report_replaced_list)
        reports_replaced_split_list.append(report_replaced_split_list)

    for i in range(len(reports_replaced_list)):
        report = reports_replaced_list[i]  
        report_word = reports_replaced_split_list[i]  

        report_organs = []
        report_diseases = []

        for j in range(len(report)): 
            sentence = report[j]
           
            sentence_word = report_word[j]
            sentence_word_set = set(sentence_word)

            sentence_organs = [] 
            sentence_diseases = [] 

            for key in disease_organ: 
                key_set = set(key.split(" "))
                if key_set.issubset(sentence_word_set):
                    sentence_organs.append(disease_organ[key]) 
                    sentence_diseases.append(key)

            sentence_organs, sentence_diseases = return_negative(  
                sentence_organs, sentence_diseases, sentence
            )

           
            sentence_organs, sentence_diseases = remove_special_case(  
                sentence_organs, sentence_diseases
            )

           
            sentence_diseases = associate_same_disease(sentence_diseases)  


            report_organs.append(sentence_organs)
            report_diseases.append(sentence_diseases)

        organs_list.append(report_organs)
        diseases_list.append(report_diseases)

    return organs_list, diseases_list



def data_to_list(datatype_str):
    flat_list = [item for sublist in datatype_str for item in sublist]
    return flat_list

def format(pair):
    parts = pair.split('-')
    if len(parts)!=2:
        print(pair)
    assert len(parts) == 2
    disease, location = parts
    formatted_pair = f"{location} shows {disease}"
    return formatted_pair

def change(key):
    for replace_key in pair_synonym:
        if replace_key==key :
            key=pair_synonym[replace_key]
        elif "no "+replace_key==key:
            key="no "+pair_synonym[replace_key]
            key=key.replace('no a ','no ').replace('no normal ','normal ')
    return key


def get_new_csv(df):
    reports_list = list(df["report"])  
    reports_sentences = divide_to_sentences(reports_list) 
    reports_clean = clean_sentence(reports_sentences) 

    df["split_by_sentence"] = reports_clean  
    organs_list, diseases_list = find_diseases_organs(reports_clean)  
    df["organs_list"] = organs_list
    df["diseases_list"] = diseases_list
    diseases_pool = {}
    for key in disease_organ:
        pool_key = key + "-" + disease_organ[key]
        no_pool_key = 'no '+key + "-" + disease_organ[key]
        pool_key = change(pool_key)
        no_pool_key = change(no_pool_key)
        diseases_pool[pool_key] = []
        diseases_pool[no_pool_key] = []

    disease_type = []
    disease_organ_pairs = []
    for i in range(len(df)):  
        item = df.iloc[i]
        organs = item["organs_list"]
        diseases = item["diseases_list"]
        sentences = item["split_by_sentence"]
        report_disease = []

        
        for j in range(len(sentences)):
            assert len(organs[j]) == len(diseases[j])
            organ = organs[j]
            disease = diseases[j]
            sentence = sentences[j]
            sentence_disease = []

            if len(organ) > 0:
                for t in range(len(organ)):
                    key = disease[t].replace('no no ','no ').replace('no no ','no ') + "-" + organ[t]
                    key=change(key)
                    sentence_disease.append(key)  

                    if sentence not in diseases_pool[key]:
                        diseases_pool[key].append(sentence)

            report_disease.append(sentence_disease)     
        disease_type.append(report_disease)
        disease_organ_pairs.append(data_to_list(report_disease))
    df["disease_type"] = disease_type
    df['pairs'] = disease_organ_pairs
    
    df.to_csv("iuxray.csv", header=True, index=False)
   


# 第一步：生成csv文件
custom_columns = ['id', 'split','image_path','report']
df = pd.DataFrame(columns=custom_columns)

with open('../meta_data/annotation.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    data_list = []
for i,section in enumerate(data.values()):  # 这将遍历每个键对应的值，即train, val, test部分
    for item in section: 
        sample = {}
        sample = {'id': item['id'], 'split':item['split'],'image_path':item['image_path'],'report': item['report']}
        data_list.append(sample)

df = pd.DataFrame(data_list)
get_new_csv(df)
print("CSV文件清洗完成")