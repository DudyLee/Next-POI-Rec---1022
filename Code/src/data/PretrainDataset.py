import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt, check_task_prompt
from utils.utils import ReadLineFromFile, load_item_info, check_item_content, sample_other_category_items, sample_higher_price_items, sample_lower_price_items, sample_lower_salesrank_items
from utils import indexing
import logging
import pdb
import re


class PretrainDataset(Dataset):
    def __init__(self, args, dataset, tasks):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = tasks.split(',')
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(args.sample_num.split(',')), "prompt sample number does not match task number"
        self.item_indexing = args.item_indexing
        self.args = args
        
        self.collaborative_token_size = args.collaborative_token_size
        self.collaborative_cluster_num = args.collaborative_cluster
        self.collaborative_last_token = args.collaborative_last_token
        self.collaborative_float32 = args.collaborative_float32
        
        # get prompt related info, including numbers and index
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)
        check_task_prompt(self.prompt, self.tasks)
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
        
        self.prefix = args.his_prefix
        
        # apply indexing method
        if self.item_indexing == 'independent':
            self.reindex_user_seq_dict, self.item_map = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'sequential':
            self.reindex_user_seq_dict, self.item_map = indexing.sequential_indexing(self.data_path, self.dataset, self.user_sequence_dict, args.sequential_order)
        elif self.item_indexing == 'random':
            self.reindex_user_seq_dict, self.item_map = indexing.random_indexing(self.data_path, self.dataset, self.user_sequence_dict)
        elif self.item_indexing == 'collaborative':
            self.reindex_user_seq_dict, self.item_map = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, \
                                                                                        self.collaborative_last_token, self.collaborative_float32)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'semantic':
                logging.info(f"Reindex data with semantic indexing method")
                self.reindex_user_seq_dict, self.item_map = indexing.semantic_indexing(self.data_path, self.dataset, self.user_sequence_dict)
        else:
            raise NotImplementedError
            
        self.all_items = list(self.item_map.values())

        # load item info
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
        # sample info
        self.sample_info_key =args.sample_info_key
        if self.sample_info_key:
            self.keep_info_key = []
            for info_key in args.keep_info_key.split(','):
                if info_key in item_info_counter:
                    self.keep_info_key.append(info_key)
                else:
                    raise ValueError(f"{info_key} is not in item info")
            self.sample_info_num = args.sample_info_num
        # create data samples
        self.data_samples = self.create_samples()
        self.get_prompt_info()
        self.construct_sentence()
    
    def get_prompt_info(self):
        """
        Calculate number of prompts and cumulative index for each task
        - task_prompt_num: save the number of prompts for each task
        - task_index: the cumulative index for each task. if task_index[i-1] <= idx < task_index[i], then the idx belongs to task[i]
            - For example, there are 100 data samples in total, there are 3 tasks, the task_prompt_num is [2,1,3], then the task_index is [200, 300, 600].
        """
        if self.args.sample_prompt == 0:
            self.task_prompt_num = [len(self.prompt[task]['seen']) for task in self.tasks]
        else:
            sample_number = self.args.sample_num.split(',')
            self.task_prompt_num = [int(sample_number[i]) for i in range(len(self.tasks))]
        self.task_index = [self.task_prompt_num[0] * len(self.data_samples)]
        for i in range(1, len(self.task_prompt_num)):
            self.task_index.append(self.task_index[i-1] + self.task_prompt_num[i] * len(self.data_samples))
        self.task_data = dict()
        for i in range(len(self.tasks)):
            if i == 0:
                start = 0
            else:
                start = self.task_index[i-1]
            end = self.task_index[i]
            task = self.tasks[i]
            self.task_data[task] = [i for i in range(start, end)]

    def create_samples(self):
        data_samples = []
        for raw_item_id, item_id in self.item_map.items():
            info = self.item_info[raw_item_id]
            if self.sample_info_key:
                raise NotImplementedError
            else:
                one_sample = dict()
                lines = []
                contents = []
                for key in self.use_info_key:
                    if key in info:
                        if key == 'categories':
                            category = info[key]
                            other_category_items = sample_other_category_items(category, self.item_map, self.item_info, 9)
                            category = [c.replace(","," ").strip().replace(" ", "_").replace("&", "and").replace("'s", "").replace("-","_") for c in category]
                            category = ".".join(category)
                            item_list_with_other_category = other_category_items+[item_id]
                            random.shuffle(item_list_with_other_category)
                            one_sample['item_list_with_other_category'] = ', '.join(item_list_with_other_category)
                            one_sample['category'] = category
                            lines.append(f'"{key}": {category}')
                        elif key == 'brand':
                            brand = info[key]
                            other_brand_items = sample_other_category_items(brand, self.item_map, self.item_info, 9)
                            item_list_with_other_brand = other_brand_items+[item_id]
                            random.shuffle(item_list_with_other_brand)
                            one_sample['item_list_with_other_brand'] = ', '.join(item_list_with_other_brand)
                            one_sample['brand'] = brand
                            lines.append(f'"{key}": "{brand}"')
                        elif key == 'price':
                            price = info[key]
                            higher_price_items = sample_higher_price_items(price, self.item_map, self.item_info, 9)
                            lower_price_items = sample_lower_price_items(price, self.item_map, self.item_info, 9)
                            item_list_with_higher_price = higher_price_items+[item_id]
                            item_list_with_lower_price = lower_price_items+[item_id]
                            random.shuffle(item_list_with_higher_price)
                            random.shuffle(item_list_with_lower_price)
                            one_sample['item_list_with_higher_price'] = ', '.join(item_list_with_higher_price)
                            one_sample['item_list_with_lower_price'] = ', '.join(item_list_with_lower_price)
                            one_sample['price'] = price
                            lines.append(f'"{key}": {price}')
                        elif key == 'description':
                            description = ' '.join(info[key].split()[:50])
                            lines.append(f'"{key}": "{description}"')
                        elif key == 'salesRank':
                            salesRank = info[key]
                            lower_salesrank_items = sample_lower_salesrank_items(salesRank, self.item_map, self.item_info, 9)
                            item_list_with_lower_salesrank = lower_salesrank_items+[item_id]
                            random.shuffle(item_list_with_lower_salesrank)
                            one_sample['item_list_with_lower_salesrank'] = ', '.join(item_list_with_lower_salesrank)
                            one_sample['salesRank'] = salesRank
                            lines.append(f'"{key}": {salesRank}')
                        else:
                            lines.append(f'"{key}": "{info[key]}"')
                        contents.append(f'{key} is "{info[key]}"')
                if len(lines) == 0:
                    continue   
                one_sample['content'] = ',\n'.join(lines)
                one_sample['language_style_content'] = '; '.join(contents)
                one_sample['item_id'] = item_id
                data_samples.append(one_sample)
        
        return data_samples
    
    def __len__(self):
        return len(self.data['input'])
    
    def construct_sentence(self):
        if self.args.sample_prompt == 0:
            self._construct_sentence_all()
        else:
            self._construct_sentence_sample()
        # logging.info(f"Input: {self.data['input'][100]}\nOutput: {self.data['output'][100]} ")
        # logging.info(f"Input: {self.data['input'][-100]}\nOutput: {self.data['output'][-100]} ")

    def _construct_sentence_all(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]['seen']:
                    try:
                        self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                        self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                    except KeyError:
                        continue
            logging.info(f"Input: {self.data['input'][-1]}\nOutput: {self.data['output'][-1]} ")
            
                    
    def _construct_sentence_sample(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]['seen']) - 1)
                    try:
                        self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                        self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
                    except KeyError:
                        continue
            logging.info(f"Input: {self.data['input'][-1]}\nOutput: {self.data['output'][-1]}\n")

    def __getitem__(self, idx):
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
        
        
        
    
    