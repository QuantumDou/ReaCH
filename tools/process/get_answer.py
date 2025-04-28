from collections import defaultdict
import ast

def clean_disease_type(pairs):
    pairs = ast.literal_eval(pairs)
    converted_pairs = []
    for pair in pairs:
        
        anomaly, organ = pair.split('-')
       
        if 'no ' not in anomaly and 'normal' not in anomaly:  
            converted_pair = f"{anomaly} in {organ}"
            converted_pairs.append(converted_pair)
    if len(converted_pairs)==0:
            return 'yes .'
    result_string = 'yes , '+' , '.join(converted_pairs)+' .'
    return result_string

def clean_normal_type(pairs):
    pairs = ast.literal_eval(pairs)
    converted_pairs = []
    for pair in pairs:
        anomaly, organ = pair.split('-')
        if 'no ' in anomaly or 'normal' in anomaly: 
            converted_pair = f"{anomaly} in {organ}"
            converted_pairs.append(converted_pair)
    if len(converted_pairs)==0:
        return 'no .'
    result_string = 'no , '+' , '.join(converted_pairs[:3])+' .'
    return result_string