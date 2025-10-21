import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt, check_task_prompt
from utils.utils import ReadLineFromFile, load_item_info, check_item_content
from utils import indexing
from collections import defaultdict
import logging
import pdb
import re


class TestDataset(Dataset):
    def __init__(self, args, dataset, task):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.item_indexing = args.item_indexing
        self.user_indexing = args.user_indexing
        
        self.collaborative_token_size = args.collaborative_token_size
        self.collaborative_cluster_num = args.collaborative_cluster
        self.collaborative_last_token = args.collaborative_last_token
        self.collaborative_float32 = args.collaborative_float32
        
        self.prompt = load_prompt_template(args.prompt_file, [self.task])
        check_task_prompt(self.prompt, [self.task])
        self.prompt_info = get_info_from_prompt(self.prompt)

        
        
        if 'history' in self.prompt_info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        if 'candidate_items' in self.prompt_info:
            self.candidate_neg_num = args.candidate_neg_num
            self.candidate_sep = args.candidate_sep
        
        # load user sequence data
        self.user_sequence = ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'sequential_data.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)

        self.max_content_num = args.max_content_num
        
        self.prefix = args.his_prefix
        
        # apply indexing method
        if self.item_indexing == 'independent' and self.user_indexing == 'independent':
            logging.info("Reindex data with independent indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.ui_independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
            for user in self.reindex_user_seq_dict:
                self.new_token += re.findall(r'\<.*?\>', user)
        elif self.item_indexing == 'independent':
            self.reindex_user_seq_dict, self.item_map = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'sequential':
            self.reindex_user_seq_dict, self.item_map = indexing.sequential_indexing(self.data_path, self.dataset, self.user_sequence_dict, args.sequential_order, False)
        elif self.item_indexing == 'random':
            self.reindex_user_seq_dict, self.item_map = indexing.random_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
        elif self.item_indexing == 'collaborative':
            self.reindex_user_seq_dict, self.item_map = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, \
                                                                                        self.collaborative_last_token, self.collaborative_float32, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'semantic':
            logging.info(f"[Test]Reindex data with semantic indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.semantic_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'collaborative_independent':
            logging.info(f"[Test]Reindex data with collaborative_independent indexing method")
            reindex_user_seq_dict_col, item_map_col = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, \
                                                                                        self.collaborative_last_token, self.collaborative_float32, False)
            reindex_user_seq_dict_ind, item_map_ind = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.reindex_user_seq_dict, self.item_map = indexing.fusion([item_map_col, item_map_ind], self.user_sequence_dict)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'semantic_independent':
            logging.info(f"[Test]Reindex data with semantic_independent method")
            reindex_user_seq_dict_sem, item_map_sem = indexing.semantic_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            reindex_user_seq_dict_ind, item_map_ind = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.reindex_user_seq_dict, self.item_map = indexing.fusion([item_map_sem, item_map_ind], self.user_sequence_dict)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        else:
            raise NotImplementedError
            
        self.all_items = list(self.item_map.values())
        self.item_map_rev = {v:k for k, v in self.item_map.items()}
        
        self.test_prompt = args.test_prompt
        self.test_filtered = args.test_filtered
        
        if args.test_filtered > 0:
            self.user2id = dict()
            self.id2user = dict()
            for user in self.reindex_user_seq_dict:
                if user not in self.user2id:
                    self.user2id[user] = len(self.user2id)
                    self.id2user[len(self.id2user)] = user
                    
            # get positive samples for each user to sample negative candidates or evaluation
            if args.test_filtered_batch > 0:
                self.positive, self.max_positive = self.get_positive_batch()
            self.positive = self.get_positive()
        
        # load item info
        if any(['content' in p for p in self.prompt_info]):
            self.item_info = load_item_info(self.data_path, self.dataset)
            # check keys in item info
            item_info_counter = check_item_content(self.item_info)
            for key, count in item_info_counter.items():
                logging.info(f"{key}: {count}")
            self.use_info_key = []
            for info_key in args.use_info_key.split(','):
                if info_key in item_info_counter:
                    self.use_info_key.append(info_key)
                else:
                    raise ValueError(f"{info_key} is not in item info")

        # load data
        self.data_samples = self.load_test()
        
        
    
        self.construct_sentence()
        # get prompt related info, including numbers and index
        # self.get_prompt_info()
        
    def load_test(self):
        """
        Load test data samples
        """
        data_samples = []
        # sort by his length
        sorted_reindex_user_seq_dict = sorted(self.reindex_user_seq_dict.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_reindex_user_seq_dict = dict(sorted_reindex_user_seq_dict)
        for user in sorted_reindex_user_seq_dict:
            items = sorted_reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample['dataset'] = self.dataset
            one_sample['user_id'] = user
            if self.prefix > 0:
                one_sample['target'] = 'item_' + items[-1]
            else:
                one_sample['target'] = items[-1]
            one_sample['target_id'] = items[-1]
            if 'history' in self.prompt_info:
                history = items[:-1]
                if self.max_his > 0:
                    history = history[-self.max_his:]
                if self.prefix > 0:
                    one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = self.his_sep.join(history)
            if 'items_content_dict' in self.prompt_info:
                items_with_content = items[:-1][-self.max_content_num:]
                contents = []
                for item_idx in items_with_content:
                    raw_idx = self.item_map_rev[item_idx]
                    content = f"item_{item_idx} = " if self.prefix else f"{item_idx} = "
                    content += "{"
                    info_lines = []
                    for key in self.use_info_key:
                        if key in self.item_info[raw_idx]:
                            if key == 'category':
                                self.item_info[item_idx][key] = '.'.join(self.item_info[raw_idx][key])
                            info_lines.append(f'"{key}": "{self.item_info[raw_idx][key]}"')
                    if len(info_lines) == 0:
                        continue
                    content += ', '.join(info_lines)
                    content += "}"
                    contents.append(content)
                one_sample['items_content_dict'] = '\n'.join(contents)
            if 'items_content_class' in self.prompt_info:
                items_with_content = items[:-1][-self.max_content_num:]
                contents = []
                for item_idx in items_with_content:
                    raw_idx = self.item_map_rev[item_idx]
                    content = f"class item_{item_idx}:\n" if self.prefix else f"class {item_idx}:\n"
                    info_lines = []
                    for key in self.use_info_key:
                        if key in self.item_info[raw_idx]:
                            if key == 'category':
                                self.item_info[item_idx][key] = '.'.join(self.item_info[raw_idx][key])
                            info_lines.append(f'\t{key} = "{self.item_info[raw_idx][key]}"')
                    if len(info_lines) == 0:
                        continue
                    content += '\n'.join(info_lines)
                one_sample['items_content_class'] = content
            data_samples.append(one_sample)
        return data_samples
    
    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.reindex_user_seq_dict:
            positive[user] = set(self.reindex_user_seq_dict[user][:-1])
            
        return positive
    
    def get_positive_batch(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        max_positive = 0
        positive = dict()
        info = self.test_prompt.split(':')
        prompt = self.prompt[self.task][info[0]][info[1]]
        data_format = dict()
        data_format['dataset'] = self.dataset
        for user in self.reindex_user_seq_dict:
            positive[user] = set()
            positive_id = self.reindex_user_seq_dict[user][:-1]
            
            for item in positive_id:
                if self.prefix > 0:
                    data_format['target'] = 'item_' + item
                else:
                    data_format['target'] = item
                positive[user].add(prompt['Output'].format(**data_format))
                
            if len(positive[user]) > max_positive:    
                max_positive = len(positive[user])
        return positive, max_positive
    
    def __len__(self):
        return len(self.data_samples)
    
    def construct_sentence(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        info = self.test_prompt.split(':')
        prompt = self.prompt[self.task][info[0]][info[1]]
        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            self.data['input'].append(prompt['Input'].format(**datapoint))
            self.data['output'].append(prompt['Output'].format(**datapoint))
        logging.info(f"Input: {self.data['input'][100]}\nOutput: {self.data['output'][100]} ")
            
    def __getitem__(self, idx):
        if self.test_filtered > 0:
            return self.get_item_filtered(idx)
        else:
            return self.get_item(idx)
        
    def get_item_filtered(self, idx):
        
        
        return {'user_idx': self.user2id[self.data_samples[idx]['user_id']],
               'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
    
    def get_item(self, idx):
        
        
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
    
    