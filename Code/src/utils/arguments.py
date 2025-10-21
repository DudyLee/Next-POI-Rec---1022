import argparse
import logging

def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoint', help='The checkpoint directory')
    parser.add_argument("--model_name", type=str, default='model.pt', help='The model name')
    parser.add_argument("--model_path", type=str, help='The model path')
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument("--distributed", type=int, default=1, help='use distributed data parallel or not.')
    parser.add_argument("--gpu", type=str, default='0,1,2,3', help='gpu ids, if not distributed, only use the first one.')
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12345', help='Setup MASTER_PORT for os.environ')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument("--debug_mode", type=int, default=0, help='debug mode or not')
    parser.add_argument("--init_new_tokens", type=int, default=0, help='initialize new tokens or not')
    return parser

def parse_dataset_args(parser):
    """
    parse dataset related command line arguments
    """
    parser.add_argument("--data_path", type=str, default='../data/', help="data directory")
    parser.add_argument("--user_indexing", type=str, default='sequential', help="user indexing method")
    parser.add_argument("--item_indexing", type=str, default='sequential', help="item indexing method")
    parser.add_argument("--tasks", type=str, default='sequential', help="Downstream tasks, separate by comma")
    parser.add_argument("--datasets", type=str, default='beauty', help="Dataset names, separate by comma")
    parser.add_argument("--prompt_file", type=str, default='../prompt.txt', help='the path of the prompt template file')
    parser.add_argument("--permutation_num", type=int, default=0)
    parser.add_argument("--random_sample_num", type=int, default=0)


    # arguments related to item indexing methods
    parser.add_argument("--sequential_order", type=str, default='original', help='The rank of user history during ')
    parser.add_argument("--collaborative_token_size", type=int, default=200, help='the number of tokens used for indexing')
    parser.add_argument("--collaborative_cluster", type=int, default=20, help='the number of clusters in each level for collaborative indexing.')
    parser.add_argument("--collaborative_last_token", type=str, default='sequential', help='how to assign the last token to items within the same clusters, random or sequential')
    parser.add_argument("--collaborative_float32", type=int, default=0, help='1 for use float32 during indexing, 0 for float64.')
    
    # arguments related to sequential task
    parser.add_argument("--max_his", type=int, default=20, help='the max number of items in history sequence, -1 means no limit')
    parser.add_argument("--his_prefix", type=int, default=0, help='whether add prefix in history')
    parser.add_argument("--his_sep", type=str, default=', ', help='The separator used for history')
    parser.add_argument("--skip_empty_his", type=int, default=1, help='whether include data with empty history.')
    
    # arguments related for evaluation
    parser.add_argument("--valid_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    parser.add_argument("--valid_prompt_sample", type=int, default=1, help='use sampled prompt for validation every epoch.')
    parser.add_argument("--valid_sample_num", type=str, default='3', help='the number of sampled data for each task')
    parser.add_argument("--test_prompt", type=str, default='seen:0', help='The prompt used for evaluation, seen/unseen: id')
    
    # arguments related to prompt sampling
    parser.add_argument("--sample_prompt", type=int, default=1, help='sample prompt or not')
    parser.add_argument("--sample_num", type=str, default='3', help='the number of sampled data for each task')

    # item content
    parser.add_argument("--max_content_num", type=int, default=5)
    return parser

def parse_sampler_args(parser):
    """
    parse sampler related command line arguments
    """
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="the batch size for evaluation")
    parser.add_argument("--dist_sampler", type=int, default=0, help='use DistributedSampler if 1, otherwise use our own sampler.')
    
    return parser

def parse_runner_args(parser):
    """
    parse dataset related command line arguments
    """
    parser.add_argument("--optim", type=str, default='adamw_hf', help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--logging_step", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--train", type=int, default=1, help='train or not')
    parser.add_argument("--backbone", type=str, default='Salesforce/codet5-small', help='backbone model name')
    # parser.add_argument("--backbone", type=str, default='Salesforce/codet5-large', help='backbone model name')
    parser.add_argument("--metrics", type=str, default='hit@1,hit@5,hit@10,ndcg@5,ndcg@10', help='Metrics used for evaluation')
    parser.add_argument("--eval_metric", type=str, default='hit@5', help='Metrics used for evaluation')
    parser.add_argument("--metric_file", type=str, default='../src/utils/metric.py')
    parser.add_argument("--filter_prediction", type=int, default=1, help='filter predicti in history')
    parser.add_argument("--load", type=int, default=0, help='load model from model path or not.')
    parser.add_argument("--random_initialize", type=int, default=0, help='Randomly initialize number-related tokens.')
    parser.add_argument("--test_epoch", type=int, default=1, help='test once for how many epochs, 0 for no test during training.')
    parser.add_argument("--valid_select", type=int, default=0, help='use validation loss to select models')
    parser.add_argument("--test_before_train", type=int, default=0, help='whether test before training')
    parser.add_argument("--test_filtered", type=int, default=0, help='whether filter out the items in the training data.')
    parser.add_argument("--test_filtered_batch", type=int, default=1, help='whether testing with filtered data in batch.')
    parser.add_argument("--use_prefix_allowed_tokens_fn", type=int, default=0, help='whether use prefix_allowed_tokens_fn')
    
    return parser

def parse_pretrain_args(parser):
    parser.add_argument("--pretrain", type=int, default=1, help='run pretrain or not')
    parser.add_argument("--use_info_key", type=str, default="title,categories,brand,price,description", help='The item info key used for pretrain, separate by ,')
    parser.add_argument("--sample_info_key", type=int, default=0, help='sample info key or not')
    parser.add_argument("--sample_info_num", type=int, default=3, help='the number of sampled info key')
    parser.add_argument("--keep_info_key", type=str, default="title,description", help='The item info key must be used for pretrain, separate by ,')
    return parser


def get_argparser():
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_runner_args(parser)
    parser = parse_sampler_args(parser)
    parser = parse_pretrain_args(parser)
    return parser