#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json
import jsonlines
import collections

import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import LOUVRERetriever, LOUVRERetrieverConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_batch_size",
                        default=1,
                        type=int)
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument("--max_q_seq_length",
                        default=70,
                        type=int)
    parser.add_argument("--max_q_sp_seq_length",
                        default=350,
                        type=int)
    parser.add_argument("--max_ctx_seq_length",
                        default=300,
                        type=int)
    parser.add_argument('--init_checkpoint',
                        type=str,
                        required=True)
    # parser.add_argument('--ctx_init_checkpoint',
    #                     type=str,
    #                     required=True)
    parser.add_argument('--topk',
                        type=int,
                        required=False)
    parser.add_argument('--beam_size',
                        type=int,
                        required=False)
    # parser.add_argument('--corpus_fname',
    #                     type=str,
    #                     required=True)
    parser.add_argument('--input_file',
                        type=str,
                        required=True)
    # parser.add_argument('--index_save_path',
    #                     type=str,
    #                     required=True)
    # parser.add_argument('--saved_index_path',
    #                     type=str,
    #                     required=True)
    parser.add_argument('--pred_save_file',
                        type=str,
                        required=True)
    args = parser.parse_args()
    return args

def set_seed(args):
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return 0

def main(args):
    _ = set_seed(args)
    model_config = \
        LOUVRERetrieverConfig( \
            init_checkpoint=args.init_checkpoint, \
            ctx_init_checkpoint=None, #args.ctx_init_checkpoint,
            max_q_seq_length= \
                200,
            max_q_sp_seq_length= \
                args.max_q_sp_seq_length,
            max_ctx_seq_length= \
                args.max_ctx_seq_length, \
            pred_batch_size= \
                args.pred_batch_size, \
            corpus_fname= \
                None, \
            index_save_path= \
                None, \
            saved_index_path= None , 
                # None if args.saved_index_path == "None" \
                #     else args.saved_index_path, \
            topk=None, # args.topk, \
            beam_size=None, #args.beam_size
                             ) 
    logger.info(model_config)
    retriever = LOUVRERetriever(model_config)
    
    with jsonlines.open(args.input_file, "r") as lines: 
        data = [ line for line in lines ] 
        
    ssg_data = retriever.pred_nldb(data)
    
    with open(args.pred_save_file, "w") as out_file:
        json.dump(ssg_data, out_file)
    
    return 0
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
