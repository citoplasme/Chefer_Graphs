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

# PLMs = {dataset : load_plm(dataset, fine_tuning_trial_number) for dataset, fine_tuning_trial_number in zip(['IMDb', 'Ohsumed', 'R8', 'SST-2'], [0, 3, 0, 0])}
PLMs = {dataset : load_plm(dataset, fine_tuning_trial_number) for dataset, fine_tuning_trial_number in zip(['Ohsumed', 'R8'], [3, 0])}


MAXIMUM_CHUNK_SIZE = 512
LEFT_STRIDE = 128
RIGHT_STRIDE = 0

LABEL_INDICES = {
  'IMDb' : [0, 1],
  'Ohsumed' : list(range(23)),
  'R8' : list(range(8)),
  'SST-2' : [0, 1],
}

SAMPLES_PER_DATASET = 5 #200

STORAGE_PATH = './statistics/'
os.makedirs(STORAGE_PATH, exist_ok = True)

tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

df = pd.concat([
  # pd.read_csv('../../data/with_validation_splits/IMDb/train.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'IMDb', split = 'train').reset_index(),
  # pd.read_csv('../../data/with_validation_splits/IMDb/validation.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'IMDb', split = 'validation').reset_index(),
  # pd.read_csv('../../data/with_validation_splits/IMDb/test.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'IMDb', split = 'test').reset_index(),
  #
  pd.read_csv('../../data/with_validation_splits/Ohsumed/train.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'Ohsumed', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/Ohsumed/validation.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'Ohsumed', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/Ohsumed/test.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'Ohsumed', split = 'test').reset_index(),
  #
  pd.read_csv('../../data/with_validation_splits/R8/train.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'R8', split = 'train').reset_index(),
  pd.read_csv('../../data/with_validation_splits/R8/validation.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'R8', split = 'validation').reset_index(),
  pd.read_csv('../../data/with_validation_splits/R8/test.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'R8', split = 'test').reset_index(),
  #
  # pd.read_csv('../../data/with_validation_splits/SST-2/train.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'SST-2', split = 'train').reset_index(),
  # pd.read_csv('../../data/with_validation_splits/SST-2/validation.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'SST-2', split = 'validation').reset_index(),
  # pd.read_csv('../../data/with_validation_splits/SST-2/test.csv').sample(SAMPLES_PER_DATASET, random_state = SEED).assign(dataset = 'SST-2', split = 'test').reset_index(),
]).reset_index(drop = True)

vectorizer = CountVectorizer()
vectorized_df = vectorizer.fit_transform(df['text'])
df['document_length'] = vectorized_df.sum(axis = 1)
df['unique_words'] = np.array((vectorized_df > 0).sum(axis = 1)).flatten()
df['token_count'] = df['text'].apply(lambda x : len(TOKENIZER.tokenize(x, add_special_tokens = True)))

df.to_csv(os.path.join(STORAGE_PATH, 'data.csv'), index = False)

# Sliding windows
sliding_window_statistics = list()
for window_size in [2, 3, 4, 5]:
  for row in df.itertuples(index = False):
    dataset = row.dataset
    split = row.split
    index = row.index
    
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
pd.DataFrame(sliding_window_statistics, columns = ['dataset', 'split', 'index', 'window_size', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'sliding_windows.csv'), index = False)

print('[UPDATE] Completed the sliding-window graph construction.', flush = True)

# Attention distillation
attention_distillation_statistics = list()
for threshold in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
  for row in df.itertuples(index = False):
    dataset = row.dataset
    split = row.split
    index = row.index

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
pd.DataFrame(attention_distillation_statistics, columns = ['dataset', 'split', 'index', 'threshold', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'attention_distillation.csv'), index = False)

print('[UPDATE] Completed the attention distillation graph construction.', flush = True)

# Chefer importance
chefer_importance_statistics = list()
for token_threshold, edge_threshold in list(itertools.product([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], repeat = 2)):
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
pd.DataFrame(chefer_importance_statistics, columns = ['dataset', 'split', 'index', 'node_threshold', 'edge_threshold', 'document_length', 'unique_words', 'token_count', 'node_count', 'node_embedding_dimension', 'edge_count', 'edge_embedding_dimension']) \
  .to_csv(os.path.join(STORAGE_PATH, 'chefer_importance.csv'), index = False)

print('[UPDATE] Completed the Chefer importance graph construction.', flush = True)