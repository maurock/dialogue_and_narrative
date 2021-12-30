# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import time
import numpy as np
import pandas as pd
import torch
import utils

split = "train"
cache_dir = "./data_cache"

dialogue_dataset = load_dataset(
    "doc2dial",
    name="dialogue_domain",  # this is the name of the dataset for the second subtask, dialog generation
    split=split,
    ignore_verifications=True,
    cache_dir=cache_dir,
)

document_dataset = load_dataset(
    "doc2dial",
    name="document_domain",  # this is the name of the dataset for the second subtask, dialog generation
    split=split,
    ignore_verifications=True,
    cache_dir=cache_dir,
)

train_dict = dict()
train_dict['train_document'] = []
train_dict['train_id_sp'] = []
train_dict['train_user_utterance'] = []
train_dict['train_doc_domain'] = []
train_dict['train_doc_id'] = []
train_dict['train_text_sp'] = []
train_dict['train_dial_id_turn_id'] = []     # necessary for evaluation
train_dict['train_start_pos'] = []     
train_dict['train_end_pos'] = []     
train_dict['train_start_tok'] = []     
train_dict['train_end_tok'] = []  

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

start = time.time()
for idx, dialogue in tqdm(enumerate(dialogue_dataset)):
    if idx == 10:
        break
    dial_id_turn_id = []       # running list of <dial_id>_<turn_id> for evaluation
    sp_id_list = []            # running list of spans per document
    user_utterance_list = []   # running list of user utterances per document
    
    for turn in dialogue['turns']:
        dial_id_turn_id.append(dialogue['dial_id'] + '_' + str(turn['turn_id']))
        if turn['role'] == 'user':
            # TURN UTTERANCE IS FLATTENED AND ONLY THE [INPUT_IDS] IS STORED
            turn['utterance'] = tokenizer(turn['utterance'], padding=True, truncation=True, return_tensors="pt")['input_ids'].view(-1)
            user_utterance_list.append(turn['utterance'])   # adding user utterance to user_utterance_list
        else:
            references = turn['references']
            ref_sp_id = []
            for ref in references:
                ref_sp_id.append(ref['sp_id'])
            sp_id_list.append(ref_sp_id)          # adding list of sp_ids per dialogue to list of sp_ids per document
    train_dict['train_id_sp'].append(sp_id_list)
    train_dict['train_user_utterance'].append(user_utterance_list)
    train_dict['train_doc_domain'].append(dialogue['domain'])
    train_dict['train_doc_id'].append(dialogue['doc_id'])
    train_dict['train_dial_id_turn_id'].append(dial_id_turn_id)
    
    for doc in document_dataset:
        if doc['doc_id'] == train_dict['train_doc_id'][-1]:
            # DOCUMENT TEXT IS NOT A TENSOR. PREVIOUSLY WE HAD tokenizer( )['index_ids'].view(-1)
            doc['doc_text'] = tokenizer(doc['doc_text'], padding=True, truncation=False, return_tensors="pt")
            train_dict['train_document'].append(doc['doc_text'])          # adding the total document text
            text_sp_2 = []            
            start_sp_list = []         # big start sp list
            end_sp_list = []           # big end sp list        
            start_tok_list = []         # big start token list
            end_tok_list = []           # big end token list     
            for train_spans_id in train_dict['train_id_sp'][-1]:    
                text_sp = ""         
                ref_start_pos_list = []
                ref_end_pos_list = []      
                for span in doc['spans']:                    
                    if span['id_sp'] in train_spans_id:
                        text_sp += span['text_sp']                        
                        ref_start_pos_list.append(span['start_sp'])
                        ref_end_pos_list.append(span['end_sp'])    
                start_pos = np.amin(ref_start_pos_list)
                start_sp_list.append(start_pos)
                # convert start_pos to start_token
                start_tok_pos = doc['doc_text'].char_to_token(start_pos)
                start_tok_list.append(start_tok_pos)
                # convert end_pos to end_token
                end_pos = np.amax(ref_end_pos_list)
                end_sp_list.append(end_pos)
                end_tok_pos = doc['doc_text'].char_to_token(end_pos)
                end_tok_list.append(end_tok_pos)
                text_sp_2.append(text_sp)
            train_dict['train_text_sp'].append(text_sp_2)
            train_dict['train_start_pos'].append(start_sp_list)
            train_dict['train_end_pos'].append(end_sp_list)
            train_dict['train_start_tok'].append(start_tok_list)
            train_dict['train_end_tok'].append(end_tok_list)
            break
end = time.time()
print(f'Total time: {end-start}')

data = pd.DataFrame(train_dict)
utils.print_train_dict(train_dict)