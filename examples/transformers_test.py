# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
import numpy as np
from towhee import ops
from statistics import mean

import os
import warnings
from transformers import logging as t_logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
t_logging.set_verbosity_error()

model_name = sys.argv[-1]
op = ops.sentence_embedding.transformers(model_name=model_name, device='cuda:3').get_op()

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = op(batch)
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    return embeddings

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STSBenchmark']
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = se.eval(transfer_tasks)
    p = []
    s = []
    for t in transfer_tasks:
        res = results[t]['all']
        p.append(res['pearson']['mean'])
        s.append(res['spearman']['mean'])
    print('pearson:', mean(p))
    print('spearman:', mean(s))
