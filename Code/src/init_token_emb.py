import torch
import os
import argparse
import logging
import transformers
from transformers import AutoTokenizer, EarlyStoppingCallback, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import ConcatDataset, DataLoader
from data.TestDataset import TestDataset
from data.PretrainDataset import PretrainDataset
from transformers import T5Config
from transformers import T5EncoderModel as P5
from utils import utils, arguments, initialization, evaluate, generation_trie
from datasets import Dataset
from rec_trainer import RecTrainer
from tqdm import tqdm

def get_pretrain_dataset(args):
    # load pretrain dataset
    datasets = args.datasets.split(',')
    tasks = args.tasks.split(',')
    if len(tasks) > 1:
        logging.warning(f"Only support single task evaluation.\nUsing {tasks[0]} task for evaluation now.")
        logging.warning(f"Only support single dataset evaluation.\nUsing {datasets[0]} dataset for evaluation now.")
    task = tasks[0]
    dataset = datasets[0]
    pretrain_data = PretrainDataset(args, dataset, task)
    return pretrain_data

def get_user_sequence(args):
    data_path = args.data_path
    dataset = args.datasets.split(',')[0]
    item_index_file = os.path.join(data_path, dataset, 'item_independent_indexing.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_independent_indexing.txt')
    user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
    item_index = utils.ReadLineFromFile(item_index_file)
    item_index_dict = {l.split()[1]: int(l.split()[0])-1 for l in item_index}
    # 判断是否保留最后两个item
    if dataset in ['NYC', 'TKY']:
        user_sequence = [list(map(item_index_dict.get, s.split()[1:])) for s in user_sequence]
    else:
        user_sequence = [list(map(item_index_dict.get, s.split()[1:-2])) for s in user_sequence]
    return user_sequence


def main(args):    
    utils.setup_logging(args)
    #utils.setup_model_path(args)
    utils.set_seed(args.seed)
    logging.info(vars(args))

    # decide output dir
    if len(args.datasets.split(',')) > 1:
        folder_name = '_'.join(args.datasets.split(','))
    else:
        folder_name = args.datasets
    output_dir = os.path.join(args.model_dir, folder_name, args.item_indexing, args.backbone)

    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        logging_strategy="epoch",
        optim='adamw_hf',
        evaluation_strategy="epoch" if args.valid_select > 0 else "no",
        dataloader_drop_last=False,
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if args.valid_select > 0 else False,
        group_by_length=False,
        predict_with_generate=False,
    )
    
    metrics = args.metrics.split(',')
    generate_num = max([int(m.split('@')[1]) for m in metrics])
    
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    pretrain_data = get_pretrain_dataset(args)
    pretrainSet = Dataset.from_list(pretrain_data)

    # load model
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError  
    model = P5.from_pretrained(args.backbone)
    model.to('cuda:0')
    # model.to('cuda:6')

    # add additional tokens and resize token embedding
    if hasattr(pretrain_data, 'new_token'):
        tokenizer.add_tokens(pretrain_data.new_token)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    
    def process_func(datapoint):
        if 't5' in args.backbone.lower():
            encoding = tokenizer(datapoint['input'], max_length=512, truncation=True)
            labels = tokenizer(datapoint['output'], max_length=512, truncation=True)
            encoding['labels'] = labels['input_ids']
        else:
            raise NotImplementedError
        return encoding
    
    # randomly initialize number related tokens
    # if args.random_initialize == 1:
    #     # logging.info("Random initialize number related tokens")
    #     utils.random_initialization(model, tokenizer, args.backbone)
    # if args.train:
    #     trainSet = trainSet.map(process_func, batched=False)
    #     if args.valid_select > 0:
    #         validSet = validSet.map(process_func, batched=False)
    pretrainSet = pretrainSet.map(process_func, batched=False)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        label_pad_token_id=-100,
    )
    embeddings = []
    with torch.no_grad():
        for data in tqdm(pretrainSet):
            input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to('cuda:0')
            outputs = model.forward(input_ids=input_ids)
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state.squeeze(0).mean(0))
    embeddings = torch.stack(embeddings) # dim = (num_token, emb_dim)
    user_sequence = get_user_sequence(args)
    for s in user_sequence:
        print("s:", s)
        print("embeddings.shape:", embeddings.shape)
        print("max(s):", max(s), "min(s):", min(s))
        user_emb = embeddings[torch.tensor(s).to('cuda:0')].mean(0)
    user_embeddings = [embeddings[torch.tensor(s).to('cuda:0')].mean(0) for s in user_sequence]
    user_embeddings = torch.stack(user_embeddings)
    # save
    torch.save(embeddings, os.path.join(args.data_path, args.datasets, 'new_token_emb.pt'))
    torch.save(user_embeddings, os.path.join(args.data_path, args.datasets, 'user_emb.pt'))
    
if __name__ == "__main__":
    parser = arguments.get_argparser()
    args, extras = parser.parse_known_args()
    main(args)
    