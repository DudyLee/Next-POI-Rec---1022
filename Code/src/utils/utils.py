import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch
import json
import gzip
from collections import Counter, defaultdict
import time
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

                        
def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict

def setup_logging(args):
    args.log_name = log_name(args)
    if len(args.datasets.split(',')) > 1:
        folder_name = '_'.join(args.datasets.split(','))
    else:
        folder_name = args.datasets
    folder = os.path.join(args.log_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = os.path.join(args.log_dir, folder_name, args.log_name + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return
    

def log_name(args):
    t = time.time()
    # format: M-D-H-M
    t_str = time.strftime("%m_%d_%H_%M", time.localtime(t))
    tasks = args.tasks.split(',')
    backbone = args.backbone.split('/')[-1] if '/' in args.backbone else args.backbone
    params = [t_str] + tasks + [backbone, args.item_indexing]
    return '_'.join(params)

def setup_model_path(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = '_'.join(args.datasets.split(','))
    else:
        folder_name = args.datasets
    if args.model_name == 'model.pt':
        model_path = os.path.join(args.model_dir, folder_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        args.model_path = os.path.join(model_path, args.log_name+'.pt')
    else:
        args.model_path = os.path.join(args.checkpoint_dir, args.model_name)
    return
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    return
    
def load_model(model, path, args, loc=None):
    if loc is None and hasattr(args, 'gpu'):
        gpuid = args.gpu.split(',')
        loc = f'cuda:{gpuid[0]}'
    state_dict = torch.load(path, map_location=loc)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_item_info(data_path, dataset):
    # load item id to asin dict
    id2item_file = os.path.join(data_path, dataset, "datamaps.json")
    with open(id2item_file, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]
    # load meta data
    meta_data, meta_dict = load_meta_data(data_path, dataset)
    num_items = len(id2item)
    print(f"Number of items: {num_items}")
    item_info = {}
    # 修改 index 的起始值
    # if dataset in ["NYC", "TKY"]:
    #     indices = range(0, num_items)
    # else:
    indices = range(1, num_items+1)
    for index in indices:
        info = find_metadata(str(index), meta_data, meta_dict, id2item)
        if dataset in ["NYC", "TKY", "NYC_20"]:
            # POI信息分支
            item_info[str(index)] = {
                "category_id": info.get("category_id", ""),
                "category_name": info.get("category_name", ""),
                "latitude": info.get("latitude", ""),
                "longitude": info.get("longitude", ""),
                "checkin_count": info.get("checkin_count", 0),
                # "tips": info.get("tips", []),
                # "tags": info.get("tags", "")
            }
        else:
            # 保留原有分支
            if dataset != "yelp":
                info["categories"] = info["categories"][0]
                if "salesRank" in info:
                    dataset_upper = dataset[0].upper() + dataset[1:]
                    if info["salesRank"] and list(info["salesRank"].keys())[0].startswith(dataset_upper):
                        info["salesRank"] = int(list(info["salesRank"].values())[0])
                    else:
                        del info["salesRank"]
                if "price" in info:
                    info["price"] = float(info["price"])
            item_info[str(index)] = info
    return item_info

def load_item_category(data_path, dataset):
    # load item id to asin dict
    id2item_file = os.path.join(data_path, dataset, "datamaps.json")
    with open(id2item_file, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]
    # load meta data
    meta_data, meta_dict = load_meta_data(data_path, dataset)
    # build category map
    category_dict, level_categories = build_category_map(dataset, meta_data, meta_dict, id2item)
    return category_dict, level_categories

def load_meta_data(data_path, dataset):
    if dataset != "yelp":
        meta_data_file = os.path.join(data_path, dataset, "meta.json.gz")
    else:
        meta_data_file = os.path.join(data_path, "yelp", "meta_data.pkl")

    if dataset != "yelp":
        meta_data = []
        for meta in parse_json_gz(meta_data_file):
            meta_data.append(meta)

        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["asin"]] = i
    else:
        with open(meta_data_file, "rb") as f:
            meta_data = pickle.load(f)
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["business_id"]] = i
    return meta_data, meta_dict

def parse_json_gz(file):
    g = gzip.open(file, "r")
    for l in g:
        yield eval(l)

def build_category_map(dataset, meta_data, meta_dict, id2item):
    category_dict = {}
    category_counts = {}
    num_items = len(id2item)
    indices = list(range(1, num_items+1))
    # sort indices by item price from low to high
    prices = []
    for i in indices:
        data = find_metadata(str(i), meta_data, meta_dict, id2item)
        if "price" in data:
            prices.append(float(data["price"]))
        else:
            prices.append(0)
    # sort by price
    indices = [x for _, x in sorted(zip(prices, indices))]
    
    # random item index by default
    #random.shuffle(indices)

    # to count in yelp time
    all_possible_categories = {}

    for i in indices:
        index = str(i)
        if dataset != "yelp":
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ][0][1:]
        else:
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ]
            if categories is None:
                categories = ""
            categories = categories.split(", ")

        # to count in yelp time
        for c in categories:
            if c not in all_possible_categories:
                all_possible_categories[c] = 1
            else:
                all_possible_categories[c] += 1
        # if no category
        if categories == [] or categories == [""]:
            categories = ["{}".format(index)]

        if dataset != "yelp":
            category_dict[index] = categories
            if tuple(categories) in category_counts:
                category_counts[tuple(categories)] += 1
            else:
                category_counts[tuple(categories)] = 1
            category_dict[index] = categories + [
                str(category_counts[tuple(categories)])
            ]

    if dataset == "yelp":
        filtered_categories = filter_categories(meta_data, meta_dict, id2item)
        for i in indices:
            index = str(i)
            # find categories in meta data
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ]
            if categories is None:
                categories = ""
            categories = categories.split(", ")
            # filter categories
            categories = {
                c: filtered_categories[c]
                for c in categories
                if c in filtered_categories.keys()
            }
            # sort categories by order
            categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            categories = [category[0] for category in categories][:5]
            # if no category
            if categories == [] or categories == [""]:
                categories = ["{}".format(i)]
            category_dict[index] = categories
            if tuple(categories) in category_counts:
                category_counts[tuple(categories)] += 1
            else:
                category_counts[tuple(categories)] = 1
            category_dict[index] = categories + [
                str(category_counts[tuple(categories)])
            ]

    max_subcategory = max([len(k) for k in list(category_counts.keys())])

    level_categories = []
    for i in range(max_subcategory):
        one_level_categories = set({})
        for categories in list(category_counts.keys()):
            if len(categories) > i:
                one_level_categories.add(categories[i])
        one_level_categories = sorted(one_level_categories)
        one_level_categories = {v: k for k, v in enumerate(list(one_level_categories))}
        level_categories.append(one_level_categories)

    return (category_dict, level_categories)

def find_metadata(index, meta_data, meta_dict, id2item):
    # print(f"Finding metadata for index: {index} \n")
    asin = id2item[index]  # index is string type
    meta_data_position = meta_dict[asin] #asin(Amazon Standard Identification Number) 
    index_meta_data = meta_data[meta_data_position]
    return index_meta_data

def filter_categories(meta_data, meta_dict, id2item):
    num_items = len(id2item)
    indices = list(range(1, num_items+1))
    random.shuffle(indices)

    all_possible_categories = []

    for i in indices:
        index = str(i)
        categories = find_metadata(index, meta_data, meta_dict, id2item)["categories"]
        if categories is None:
            categories = ""
        categories = categories.split(", ")

        all_possible_categories += categories

    all_possible_categories = Counter(all_possible_categories)

    all_possible_categories = {
        k: v for k, v in all_possible_categories.items() if v > 10
    }

    return all_possible_categories

def check_item_content(item_info):
    # count keys
    counter = defaultdict(int)
    counter['number_of_items'] = len(item_info)
    for item_id, info_dict in item_info.items():
        for k in info_dict:
            counter[k] += 1
    return counter

def extract_history(text, sep=', '):
    """extract between []"""
    if text.startswith('#'):
        history = re.findall(r'\[(.*?)\]', text)[0]
        return history.split(sep)
    else:
        return re.findall(r'item_[\w]+', text)


def hit_count(history, predictions, max_num=-1):
    if max_num > 0:
        predictions = predictions[:max_num]
    hit = 0
    for p in predictions:
        if len(p.split(' ')) == 2:
            p = p.split(' ')[1]
        if p in history:
            hit += 1
    return hit

def sample_other_category_items(category, item_map, item_info, num_items=9):
    other_items = []
    count = 0
    while count < num_items:
        i = random.choice(list(item_map.keys()))
        if 'categories' in item_info[i] and item_info[i]['categories'] != category:
            other_items.append(item_map[i])
            count += 1
    return other_items

def sample_other_brand_items(brand, item_map, item_info, num_items=9):
    other_items = []
    count = 0
    while count < num_items:
        i = random.choice(list(item_map.keys()))
        if 'brand' in item_info[i] and item_info[i]['brand'] != brand:
            other_items.append(item_map[i])
            count += 1
    return other_items

def sample_higher_price_items(price, item_map, item_info, num_items=9):
    other_items = []
    count = 0
    patient = 0
    while count < num_items and patient < 50:
        i = random.choice(list(item_map.keys()))
        if 'price' in item_info[i] and item_info[i]['price'] > price:
            other_items.append(item_map[i])
            count += 1
        patient += 1
    return other_items

def sample_lower_price_items(price, item_map, item_info, num_items=9):
    other_items = []
    count = 0
    patient = 0
    while count < num_items and patient < 50:
        i = random.choice(list(item_map.keys()))
        if 'price' in item_info[i] and float(item_info[i]['price']) < price:
            other_items.append(item_map[i])
            count += 1
        patient += 1
    return other_items

def sample_lower_salesrank_items(salesrank, item_map, item_info, num_items=9):
    other_items = []
    count = 0
    patient = 0
    while count < num_items and patient < 50:
        i = random.choice(list(item_map.keys()))
        if 'salesRank' in item_info[i] and item_info[i]['salesRank'] > salesrank:
            other_items.append(item_map[i])
            count += 1
        patient += 1
    return other_items
