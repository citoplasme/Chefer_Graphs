import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import transformers
import construct

import numpy as np
import random
import gc

import time

# import torch_geometric
# import shutil

random_state = 42
gc.collect()
torch.cuda.empty_cache()

random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(random_state)
torch.use_deterministic_algorithms(mode = True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
  MODEL_NAME = 'textattack/bert-base-uncased-SST-2'
  
  # TEXT = 'Average movie.'
  # TEXT = 'Awful movie.'
  # TEXT = 'Awesome movie.'
  # TEXT = 'Although I laughed a lot, this has to be one of the worst movies ever made!'
  TEXT = 'I enjoyed the acting, but the story was confusing and the ending left me disappointed.'
  print('TEXT:', TEXT, flush = True)

  tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
  model = transformers.AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    output_attentions = True,
    attn_implementation = 'eager',
    output_hidden_states = True,
  )
  model.to(DEVICE)
  model.eval()

  st = time.time()
  document_level_graph = construct.construct_PyG_graph_from_Chefer_importance(
    text = TEXT,
    label = 0,
    index = 110,
    #
    label_indices = [0, 1],
    #
    chunk_size = 512,
    left_stride = 128,
    right_stride = 0,
    #
    token_threshold = 0.85,
    edge_threshold = 0.75,
    #
    plm = model,
    tokenizer = tokenizer,
    maximum_chunk_size = 512,
    #
    drop_first_level_edges_ablation = False,
    unit_weight_edges_ablation = False,
    drop_second_level_nodes_ablation = False,
    drop_second_and_third_level_nodes_ablation = False,
    #
    device = DEVICE,
  )
  print('Elapsed time:', time.time() - st, flush = True)

  print('', flush = True)
  print('Tokens/nodes:', document_level_graph['tokens'], flush = True)
  print('Node attributes:', document_level_graph['x'], flush = True)
  # print('Node attributes:', flush = True)
  # for x in document_level_graph['x']:
  #   print(x[:5].tolist(), '...', x[-5:].tolist(), flush = True)
  print('Edge identifiers:', document_level_graph['edge_index'], flush = True)
  print('Edge weights:', document_level_graph['edge_attr'], flush = True)
  print('Document label:', document_level_graph['y'], flush = True)
  print('Document identifier:', document_level_graph['identifier'], flush = True)

  # # Store in disk
  # graph_path = os.path.join('.', f'{document_level_graph["identifier"].item()}.pt')
  # torch.save(document_level_graph, graph_path)

  # # Load from disk
  # if os.path.isfile(graph_path):
  #   graph_properties = torch.load(graph_path)
  #   # del graph_properties['tokens']
  #   PyG_graph = torch_geometric.data.Data(**graph_properties)
  #   print(PyG_graph, flush = True)
  #   print('', flush = True)
  #   print('Tokens/nodes:', PyG_graph['tokens'], flush = True)
  #   print('Node attributes:', PyG_graph['x'], flush = True)
  #   print('Edge identifiers:', PyG_graph['edge_index'], flush = True)
  #   print('Edge weights:', PyG_graph['edge_attr'], flush = True)
  #   print('Document label:', PyG_graph['y'], flush = True)
  #   print('Document identifier:', PyG_graph['identifier'], flush = True)
  #   # shutil.rmtree(graph_path) # for directories
  #   os.remove(graph_path)
  # else:
  #   print(f'{graph_path} does not exist.', flush = True)
