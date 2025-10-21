import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.prompt import load_prompt_template, get_info_from_prompt,check_task_prompt
from utils.utils import ReadLineFromFile, load_item_info, check_item_content
from utils import indexing
from collections import defaultdict
import torch.distributed as dist
import logging
import re


class MultiTaskDataset(Dataset):
    def __init__(self, args, dataset, mode):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(',')
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(args.sample_num.split(',')), "prompt sample number does not match task number"
        self.item_indexing = args.item_indexing
        self.user_indexing = args.user_indexing
        self.mode = mode
        self.args = args
        self.max_content_num = args.max_content_num
        self.permutation_num = args.permutation_num
        self.random_sample_num = args.random_sample_num
        
        self.rank = 0
        self.prefix = args.his_prefix
        self.skip_empty_his = args.skip_empty_his
        self.collaborative_token_size = self.args.collaborative_token_size
        self.collaborative_cluster_num = self.args.collaborative_cluster
        self.collaborative_last_token = self.args.collaborative_last_token
        self.collaborative_float32 = self.args.collaborative_float32
        
        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")
        
        # load and check prompt
        if self.rank == 0:
            logging.info(f"Get prompt template from {args.prompt_file}")
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)
        if self.rank == 0:
            logging.info(f"{self.prompt[self.tasks[0]]['seen']['0']['Input']}")
        check_task_prompt(self.prompt, self.tasks)
        self.prompt_info = get_info_from_prompt(self.prompt)
        if self.rank == 0:
            logging.info(f"Required info: {self.prompt_info}")
        
        if 'history' in self.prompt_info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep
        
        # load user sequence data
        self.user_sequence = ReadLineFromFile(os.path.join(self.data_path, self.dataset, 'sequential_data.txt'))
        self.user_sequence_dict = indexing.construct_user_sequence_dict(self.user_sequence)
        
        if self.item_indexing == 'independent' and self.user_indexing == 'independent':
            logging.info(f"[{mode}]"+"Reindex data with independent indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.ui_independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
            for user in self.reindex_user_seq_dict:
                self.new_token += re.findall(r'\<.*?\>', user)
        elif self.item_indexing == 'independent':
            logging.info(f"[{mode}]"+"Reindex data with independent indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'sequential':
            logging.info(f"[{mode}]"+"Reindex data with sequential indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.sequential_indexing(self.data_path, self.dataset, self.user_sequence_dict, args.sequential_order, False)
        elif self.item_indexing == 'random':
            logging.info(f"[{mode}]"+"Reindex data with random indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.random_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
        elif self.item_indexing == 'collaborative':
            logging.info(f"[{mode}]"+f"Reindex data with collaborative indexing method with token_size {self.collaborative_token_size} and {self.collaborative_cluster_num} cluster")
            self.reindex_user_seq_dict, self.item_map = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, \
                                                                                        self.collaborative_last_token, self.collaborative_float32, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'semantic':
            logging.info(f"[{mode}]"+f"Reindex data with semantic indexing method")
            self.reindex_user_seq_dict, self.item_map = indexing.semantic_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'collaborative_independent':
            logging.info(f"[{mode}]"+f"Reindex data with collaborative_independent indexing method")
            reindex_user_seq_dict_col, item_map_col = indexing.collaborative_indexing(self.data_path, self.dataset, self.user_sequence_dict, \
                                                                                        self.collaborative_token_size, self.collaborative_cluster_num, \
                                                                                        self.collaborative_last_token, self.collaborative_float32, False)
            reindex_user_seq_dict_ind, item_map_ind = indexing.independent_indexing(self.data_path, self.dataset, self.user_sequence_dict, False)
            self.reindex_user_seq_dict, self.item_map = indexing.fusion([item_map_col, item_map_ind], self.user_sequence_dict)
            self.new_token = []
            for idx in list(self.item_map.values()):
                self.new_token += re.findall(r'\<.*?\>', idx)
        elif self.item_indexing == 'semantic_independent':
            logging.info(f"[{mode}]"+"Reindex data with semantic_independent method")
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
            
        # get positive samples for each user to sample negative candidates or evaluation
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
        if self.mode == 'train':
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()
        elif self.mode == 'validation':
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
            self.valid_prompt = args.valid_prompt
            if self.rank == 0:
                logging.info(f"The validation prompt is {self.valid_prompt}")
        else:
            raise NotImplementedError
            
            
        # get prompt related info, including numbers and index
        self.get_prompt_info()
        
        self.construct_sentence()
    
    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.reindex_user_seq_dict:
            if self.mode == 'train':
                positive[user] = set(self.reindex_user_seq_dict[user][:-2])
            if self.mode == 'validation':
                positive[user] = set(self.reindex_user_seq_dict[user][:-1])
            if self.mode == 'test':
                positive[user] = set(self.reindex_user_seq_dict[user])
        return positive
    
    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        
        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]
        
        
    def get_prompt_info(self):
        """
        Calculate number of prompts and cumulative index for each task
        - task_prompt_num: save the number of prompts for each task
        - task_index: the cumulative index for each task. if task_index[i-1] <= idx < task_index[i], then the idx belongs to task[i]
            - For example, there are 100 data samples in total, there are 3 tasks, the task_prompt_num is [2,1,3], then the task_index is [200, 300, 600].
        """
        if self.rank == 0:
            logging.info(f"Getting prompt information")
        if self.mode == 'train':
            if self.args.sample_prompt == 0:
                self.task_prompt_num = [len(self.prompt[task]['seen']) for task in self.tasks]
            else:
                sample_number = self.args.sample_num.split(',')
                self.task_prompt_num = [int(sample_number[i]) for i in range(len(self.tasks))]
        else:
            if self.args.valid_prompt_sample == 0:
                self.task_prompt_num = [1] * len(self.tasks)
            else:
                sample_number = self.args.valid_sample_num.split(',')
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
    
    def load_train(self):
        """
        Load training data samples
        """
        data_samples = []
        print(f"开始遍历用户，总用户数: {len(self.reindex_user_seq_dict)}")
        for user_idx, user in enumerate(self.reindex_user_seq_dict):
            if user_idx % 1000 == 0:
                print(f"处理到第 {user_idx} 个用户: {user}")
            items = self.reindex_user_seq_dict[user][:-2]
            print(f"用户 {user} 的序列长度: {len(items)}")
            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample['dataset'] = self.dataset
                one_sample['user_id'] = user
                if self.prefix > 0:
                    one_sample['target'] = 'item_' + items[i]
                else:
                    one_sample['target'] = items[i]
                one_sample['target_id'] = items[i]
                if 'history' in self.prompt_info:
                    history = items[:i]
                    if self.max_his > 0:
                        history = history[-self.max_his:]
                    if self.prefix > 0:
                        one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                    else:
                        one_sample['history'] = self.his_sep.join(history)
                if 'items_content_dict' in self.prompt_info:
                    items_with_content = items[:i][-self.max_content_num:]
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
                    items_with_content = items[:i][-self.max_content_num:]
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
            if user_idx % 1000 == 0:
                print(f"已完成用户 {user_idx} 的所有样本生成，当前总样本数: {len(data_samples)}")
        print(f"所有用户处理完毕，总样本数: {len(data_samples)}")
        return data_samples
    
    def load_validation(self):
        """
        Load validation data samples
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
                one_sample['target'] = 'item_' + items[-2]
            else:
                one_sample['target'] = items[-2]
            one_sample['target_id'] = items[-2]
            if 'history' in self.prompt_info:
                history = items[:-2]
                if self.max_his > 0:
                    history = history[-self.max_his:]
                if self.prefix > 0:
                    one_sample['history'] = self.his_sep.join(["item_" + item_idx for item_idx in history])
                else:
                    one_sample['history'] = self.his_sep.join(history)
            if 'items_content_dict' in self.prompt_info:
                items_with_content = items[:-2][-self.max_content_num:]
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
                    content += ',\n'.join(info_lines)
                    content += "}"
                    contents.append(content)
                one_sample['items_content_dict'] = '\n'.join(contents)
            if 'items_content_class' in self.prompt_info:
                items_with_content = items[:-2][-self.max_content_num:]
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
    
        
    def __len__(self):
        return len(self.data['input'])
    
    def construct_sentence(self):
        if self.mode == 'train':
            if self.args.sample_prompt == 0:
                self._construct_sentence_all()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(f"Input: {self.data['input'][100]}\nOutput: {self.data['output'][100]} ")
        elif self.mode == 'validation':
            if self.args.valid_prompt_sample == 0:
                self._construct_sentence_valid()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(f"Input: {self.data['input'][100]}\nOutput: {self.data['output'][100]} ")
    
    def _construct_sentence_valid(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        setting = self.valid_prompt.split(':')
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                self.data['input'].append(self.prompt[task][setting[0]][setting[1]]['Input'].format(**datapoint))
                self.data['output'].append(self.prompt[task][setting[0]][setting[1]]['Output'].format(**datapoint))
    
    def _construct_sentence_all(self):
        self.data = {}
        self.data['input'] = []
        self.data['output'] = []
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]['seen']:
                    self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                history = datapoint['history'].split(self.his_sep)
                for j in range(self.permutation_num):
                    if len(history) > 1:
                        permuted_history = random.sample(history, len(history))
                        datapoint['history'] = self.his_sep.join(permuted_history)
                        self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                        self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))
                for s in range(self.random_sample_num):
                    if len(history) > 1:
                        sampled_history = random.sample(history, random.randint(1, len(history)))
                        datapoint['history'] = self.his_sep.join(sampled_history)
                        self.data['input'].append(self.prompt[task]['seen'][pid]['Input'].format(**datapoint))
                        self.data['output'].append(self.prompt[task]['seen'][pid]['Output'].format(**datapoint))

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
                    self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                    self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
                    history = datapoint['history'].split(self.his_sep)
                    for p in range(self.permutation_num):
                        if len(history) > 1:
                            permuted_history = random.sample(history, len(history))
                            datapoint['history'] = self.his_sep.join(permuted_history)
                            self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                            self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
                    for s in range(self.random_sample_num):
                        if len(history) > 1:
                            sampled_history = random.sample(history, random.randint(1, len(history)))
                            datapoint['history'] = self.his_sep.join(sampled_history)
                            self.data['input'].append(self.prompt[task]['seen'][str(pid)]['Input'].format(**datapoint))
                            self.data['output'].append(self.prompt[task]['seen'][str(pid)]['Output'].format(**datapoint))
    
    def __getitem__(self, idx):
        # data_id, prompt = self.identify_prompt(idx)
        # datapoint = self.data_samples[data_id]
        
        # return {'input': prompt['Input'].format(**datapoint),
        #        'output': prompt['Output'].format(**datapoint)}
        
        return {'input': self.data['input'][idx],
               'output': self.data['output'][idx]}
