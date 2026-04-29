import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import transformers
import numpy as np
import importlib.util
import sys
import torch
import random
import gc

def load_module(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module

# sliding_windows = load_module('sliding_windows_construct', '../../baselines/sliding_windows/construct.py')
# attention_distillation = load_module('attention_distillation_construct', '../../baselines/raw_attention_distillation/construct.py')
import sliding_windows
import raw_attention_distillation as attention_distillation

chefer = load_module('chefer_importance', '../../graph_construction/chefer_importance.py')
chefer_importance = load_module('chefer_importance_construct', '../../graph_construction/construct.py')

SEED = 42

gc.collect()
torch.cuda.empty_cache()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(SEED)
torch.use_deterministic_algorithms(mode = True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKENIZER = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

def load_plm(dataset, fine_tuning_trial_number):
  return transformers.AutoModelForSequenceClassification.from_pretrained(
    os.path.join('..', '..', 'fine_tuning', 'models', dataset, f'{fine_tuning_trial_number}', f'{SEED}'),
    output_attentions = True,
    attn_implementation = 'eager',
    output_hidden_states = True,
  )

# PLMs = {dataset : load_plm(dataset, fine_tuning_trial_number) for dataset, fine_tuning_trial_number in zip(['IMDb-1k', 'Ohsumed', 'R8', 'SST-2'], [56, 0, 0, 104])}
PLMs = {dataset : load_plm(dataset, fine_tuning_trial_number) for dataset, fine_tuning_trial_number in zip(['SST-2'], [104])}

MAXIMUM_CHUNK_SIZE = 512
LEFT_STRIDE = 128
RIGHT_STRIDE = 0

LABEL_INDICES = {
  'IMDb-1k' : [0, 1],
  'Ohsumed' : list(range(23)),
  'R8' : list(range(8)),
  'SST-2' : [0, 1],
}

SAMPLES_PER_DATASET = {
  'SST-2' : {
    'test' : [16, 17, 18, 92, 111, 143, 224, 338]
  },
}

CHEFER_THRESHOLDS = {
  'SST-2' : (0.75, 0.7),
}

STORAGE_PATH = './graphs/'
os.makedirs(STORAGE_PATH, exist_ok = True)

tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

df = pd.concat([
  pd.read_csv('../../data/with_validation_splits/SST-2/test.csv').iloc[SAMPLES_PER_DATASET['SST-2']['test']].assign(dataset = 'SST-2', split = 'test').reset_index(),
]).reset_index(drop = True)

vectorizer = CountVectorizer()
vectorized_df = vectorizer.fit_transform(df['text'])
df['document_length'] = vectorized_df.sum(axis = 1)
df['unique_words'] = np.array((vectorized_df > 0).sum(axis = 1)).flatten()
df['token_count'] = df['text'].apply(lambda x : len(TOKENIZER.tokenize(x, add_special_tokens = True)))

df.to_csv(os.path.join(STORAGE_PATH, 'data-looser_threshold.csv'), index = False)

# Chefer importance
os.makedirs(os.path.join(STORAGE_PATH, 'chefer_importance-looser_threshold'), exist_ok = True)
for row in df.itertuples(index = False):
  dataset = row.dataset
  split = row.split
  index = row.index
  
  PLM = PLMs[dataset].to(DEVICE)

  graph = document_level_graph = chefer_importance.construct_PyG_graph_from_Chefer_importance(
    text = row.text,
    label = row.label,
    index = index,
    #
    label_indices = LABEL_INDICES[dataset],
    #
    chunk_size = MAXIMUM_CHUNK_SIZE,
    left_stride = LEFT_STRIDE,
    right_stride = RIGHT_STRIDE,
    #
    token_threshold = CHEFER_THRESHOLDS[dataset][0],
    edge_threshold = CHEFER_THRESHOLDS[dataset][1],
    #
    plm = PLM,
    tokenizer = TOKENIZER,
    maximum_chunk_size = MAXIMUM_CHUNK_SIZE,
    #
    drop_first_level_edges_ablation = False,
    unit_weight_edges_ablation = False,
    drop_second_level_nodes_ablation = False,
    drop_second_and_third_level_nodes_ablation = False,
    bi_directional_edges_to_second_and_third_level_nodes_ablation = False,
    #
    device = DEVICE
  )
  torch.save(graph, os.path.join(STORAGE_PATH, 'chefer_importance-looser_threshold', f'{dataset}-{split}-{index}.pt'))
  del graph

  PLM.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()
print('[UPDATE] Completed the Chefer importance graph construction.', flush = True)