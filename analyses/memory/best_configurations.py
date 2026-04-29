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

sliding_windows = load_module('sliding_windows_construct', '../../baselines/sliding_windows/construct.py')
attention_distillation = load_module('attention_distillation_construct', '../../baselines/raw_attention_distillation/construct.py')
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

PLMs = {dataset : load_plm(dataset, fine_tuning_trial_number) for dataset, fine_tuning_trial_number in zip(['IMDb-1k', 'Ohsumed', 'R8', 'SST-2'], [56, 0, 0, 104])}

MAXIMUM_CHUNK_SIZE = 512
LEFT_STRIDE = 128
RIGHT_STRIDE = 0

LABEL_INDICES = {
  'IMDb-1k' : [0, 1],
  'Ohsumed' : list(range(23)),
  'R8' : list(range(8)),
  'SST-2' : [0, 1],
}

STORAGE_PATH = './statistics/best_configurations/'
os.makedirs(STORAGE_PATH, exist_ok = True)

tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

df = pd.concat([
  pd.read_csv('../../data/with_validation_splits/IMDb-1k/train.csv').assign(dataset = 'IMDb-1k', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/IMDb-1k/validation.csv').assign(dataset = 'IMDb-1k', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/IMDb-1k/test.csv').assign(dataset = 'IMDb-1k', split = 'test').reset_index(),
  #
  pd.read_csv('../../data/with_validation_splits/Ohsumed/train.csv').assign(dataset = 'Ohsumed', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/Ohsumed/validation.csv').assign(dataset = 'Ohsumed', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/Ohsumed/test.csv').assign(dataset = 'Ohsumed', split = 'test').reset_index(),
  #
  pd.read_csv('../../data/with_validation_splits/R8/train.csv').assign(dataset = 'R8', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/R8/validation.csv').assign(dataset = 'R8', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/R8/test.csv').assign(dataset = 'R8', split = 'test').reset_index(),
  #
  pd.read_csv('../../data/with_validation_splits/SST-2/train.csv').assign(dataset = 'SST-2', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/SST-2/validation.csv').assign(dataset = 'SST-2', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/SST-2/test.csv').assign(dataset = 'SST-2', split = 'test').reset_index(),
]).reset_index(drop = True)

vectorizer = CountVectorizer()
vectorized_df = vectorizer.fit_transform(df['text'])
df['document_length'] = vectorized_df.sum(axis = 1)
df['unique_words'] = np.array((vectorized_df > 0).sum(axis = 1)).flatten()
df['token_count'] = df['text'].apply(lambda x : len(TOKENIZER.tokenize(x, add_special_tokens = True)))

# Sliding windows
best_sliding_window_per_dataset = {
  'SST-2' : 3,
  'R8' : 4,
  'Ohsumed' : 3,
  'IMDb-1k' : 2
}
sliding_window_statistics = list()
for row in df.itertuples(index = False):
  dataset = row.dataset
  split = row.split
  index = row.index
  window_size = best_sliding_window_per_dataset[dataset]
  
  PLM = PLMs[dataset].to(DEVICE)
  
  graph = sliding_windows.construct_PyG_graph_from_sliding_windows(
    text = row.text,
    label = row.label,
    index = index,
    classification = True,
    #
    chunk_size = MAXIMUM_CHUNK_SIZE,
    left_stride = LEFT_STRIDE,
    right_stride = RIGHT_STRIDE,
    #
    surrogate = False,
    embedding_pooling = 'mean',
    window_size = window_size,
    co_occurrence_pooling = 'sum',
    #
    plm = PLM,
    tokenizer = TOKENIZER,
    maximum_chunk_size = MAXIMUM_CHUNK_SIZE,
    #
    device = DEVICE
  )

  PLM.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()
  
  document_length = row.document_length
  unique_words = row.unique_words
  token_count = row.token_count
  node_count = graph['x'].size(0)
  node_embedding_dimension = graph['x'].size(1)
  edge_count = graph['edge_attr'].size(0)
  edge_embedding_dimension = graph['edge_attr'].size(1)
  
  sliding_window_statistics.append((dataset, split, index, window_size, document_length, unique_words, token_count, node_count, node_embedding_dimension, edge_count, edge_embedding_dimension))
  del graph
pd.DataFrame(sliding_window_statistics, columns = ['dataset', 'split', 'index', 'window_size', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'sliding_windows.csv'), index = False)

print('[UPDATE] Completed the sliding-window graph construction.', flush = True)

# Attention distillation
best_attention_threshold_per_dataset = {
  'SST-2' : 0.6,
  'R8' : 0.9299999999999999,
  'Ohsumed' : 0.82,
  'IMDb-1k' : 0.95
}
attention_distillation_statistics = list()
for row in df.itertuples(index = False):
  dataset = row.dataset
  split = row.split
  index = row.index
  threshold = best_attention_threshold_per_dataset[dataset]

  PLM = PLMs[dataset].to(DEVICE)

  graph = attention_distillation.construct_PyG_graph_from_raw_attentions(
    text = row.text,
    label = row.label,
    index = index,
    #
    classification = True,
    #
    chunk_size = MAXIMUM_CHUNK_SIZE,
    left_stride = LEFT_STRIDE,
    right_stride = RIGHT_STRIDE,
    #
    surrogate = True,
    attention_pooling = 'mean',
    embedding_pooling = 'mean',
    threshold = threshold,
    aggregation_level = 1,
    #
    plm = PLM,
    tokenizer = TOKENIZER,
    attention_output_key = 'attentions',
    maximum_chunk_size = MAXIMUM_CHUNK_SIZE,
    layers = PLM.config.num_hidden_layers,
    heads = PLM.config.num_attention_heads,
    #
    device = DEVICE
  )

  PLM.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()
  
  document_length = row.document_length
  unique_words = row.unique_words
  token_count = row.token_count
  node_count = graph['x'].size(0)
  node_embedding_dimension = graph['x'].size(1)
  edge_count = graph['edge_attr'].size(0)
  edge_embedding_dimension = graph['edge_attr'].size(1)
  
  attention_distillation_statistics.append((dataset, split, index, threshold, document_length, unique_words, token_count, node_count, node_embedding_dimension, edge_count, edge_embedding_dimension))
  del graph
pd.DataFrame(attention_distillation_statistics, columns = ['dataset', 'split', 'index', 'threshold', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'attention_distillation.csv'), index = False)

print('[UPDATE] Completed the attention distillation graph construction.', flush = True)

# Chefer importance
best_thresholds_per_dataset = {
  'SST-2' : {
    'node' : 0.95,
    'edge' : 0.69
  },
  'R8' : {
    'node' : 0.8099999999999999,
    'edge' : 0.58
  },
  'Ohsumed' : {
    'node' : 0.82,
    'edge' : 0.72
  },
  'IMDb-1k' : {
    'node' : 0.9,
    'edge' : 0.6
  }
}
chefer_importance_statistics = list()
for row in df.itertuples(index = False):
  dataset = row.dataset
  split = row.split
  index = row.index
  token_threshold = best_thresholds_per_dataset[dataset]['node']
  edge_threshold = best_thresholds_per_dataset[dataset]['edge']
  
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
    token_threshold = token_threshold,
    edge_threshold = edge_threshold,
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

  PLM.to('cpu')
  gc.collect()
  torch.cuda.empty_cache()
  
  document_length = row.document_length
  unique_words = row.unique_words
  token_count = row.token_count
  node_count = graph['x'].size(0)
  node_embedding_dimension = graph['x'].size(1)
  edge_count = graph['edge_attr'].size(0)
  edge_embedding_dimension = graph['edge_attr'].size(1)

  chefer_importance_statistics.append((dataset, split, index, token_threshold, edge_threshold, document_length, unique_words, token_count, node_count, node_embedding_dimension, edge_count, edge_embedding_dimension))
  del graph
pd.DataFrame(chefer_importance_statistics, columns = ['dataset', 'split', 'index', 'node_threshold', 'edge_threshold', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'chefer_importance.csv'), index = False)

print('[UPDATE] Completed the Chefer importance graph construction.', flush = True)