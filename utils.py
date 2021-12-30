#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:06:51 2021

@author: ri21540
"""
import time

def print_train_dict(train_dict):
    print('User utterances:')
    print(train_dict['train_user_utterance'][0])
    
    print('\nID Sp:')
    print(train_dict['train_id_sp'][0])
    
    print('\nDoc ID:')
    print(train_dict['train_doc_id'][0])
    
    print('\nDoc domain:')
    print(train_dict['train_doc_domain'][0])
    
    print('\nTrain text spans:')
    print(train_dict['train_text_sp'][0])
    
    print('\nDial_ID Turn_ID:')
    print(train_dict['train_dial_id_turn_id'][0])
    
    print('\nStart span pos:')
    print(train_dict['train_start_pos'][0])
    
    print('\nEnd span pos:')
    print(train_dict['train_end_pos'][0])
    
    print('\nStart span token:')
    print(train_dict['train_start_tok'][0])
    
    print('\nEnd span token:')
    print(train_dict['train_end_tok'][0])
    

def text_from_spans(search_domain, search_doc_id, search_id_sp, document_dataset):
    """Return text from span given doc_id (e.g. 'Top 5 DMV Mistakes and How to Avoid Them#3_0')
    and search domain (e.g. dmv)
    search_id_sp = list of strings, e.g. ['56']
    search_domain = str
    search_doc_id = str """
    start = time.time()
    total_answer = ''
    for idx, doc in enumerate(document_dataset):
        if doc['domain'] == search_domain and doc['doc_id'] == search_doc_id:
            for span in doc['spans']:
                if span['id_sp'] in search_id_sp:
                    total_answer+=span['text_sp']
            break
    print(f"Time elapsed: {time.time() - start}")
    return total_answer, idx