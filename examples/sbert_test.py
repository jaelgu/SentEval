# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Clone GenSen repo here: https://github.com/Maluuba/gensen.git
And follow instructions for loading the model used in batcher
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
import numpy as np
from towhee import ops
from statistics import mean

import os
import warnings
from transformers import logging as t_logging

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--code_dir', type=str, default='../',
                    help='It is the path of SenEval codes.')
parser.add_argument('--data_dir', type=str, default='../data',
                    help='It is the path of data.')
parser.add_argument('--model', type=str, required=True, help='It is the model name used in sbert.')
parser.add_argument('--device', type=int, default=-1, help='It is the cuda id to be used. The default value is -1, which will use cpu instead.')
parser.add_argument('--tasks', nargs='+', default=['STSBenchmark'], help='It is task type. You can add task types such as STSBenchmark, STSBenchmark-CN.')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
t_logging.set_verbosity_error()

# Set PATHs
PATH_TO_SENTEVAL = args.code_dir
PATH_TO_DATA = args.data_dir
MODEL_NAME = args.model
DEVICE = 'cpu' if args.device == -1 else f'cuda:{args.device}'
TASKS = args.tasks

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from senteval.utils import cosine

# op = ops.sentence_embedding.transformers(model_name=model_name, device='cuda:3').get_op()
op = ops.sentence_embedding.sbert(model_name=MODEL_NAME, device=DEVICE).get_op()
dim = op('test').shape[0]

# SentEval prepare and batcher
# def prepare(params, samples):
#     return 

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = op(batch)
    return np.vstack(embeddings)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
#                                  'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher)
    # transfer_tasks = ['STSBenchmark']
    transfer_tasks = TASKS
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = se.eval(transfer_tasks)
    # for k, v in results.items():
    #     print(f'\n{k}:')
    #     print('dim:', dim)
    #     print('pearson:', v['pearson'])
    #     print('spearman:', v['spearman'])
    #     print('mse:', v['mse'])
    p = []
    s = []
    for t in transfer_tasks:
        test_keys = []
        for k in results[t].keys():
            if 'test' in k:
                test_keys.append(k)
        assert len(test_keys) == 1, f'More than 1 test sets: {test_keys}'
        res = results[t][test_keys[0]]
        count = res['nsamples']
        _pearson = res['pearson'].correlation
        _spearman = res['spearman'].correlation
        print('\n{} with size {}: pearson-{:.3f}, spearman-{:.3f}'.format(t, count, _pearson, _spearman))
        p.append(_pearson)
        s.append(_spearman)
    print('\nmean pearson:', mean(p))
    print('mean spearman:', mean(s))
