import os
import torch
import torch_geometric
import gc

import construct

class CachedGraphDataset(torch_geometric.data.Dataset):
  def __init__(
    self,
    data_path,
    graph_count,
  ):
    super().__init__()
    self.data_path = data_path
    self.graph_count = graph_count
  
  def len(self):
    return self.graph_count

  def get(self, idx):
    graph_path = os.path.join(self.data_path, f'{idx}.pt')
    if os.path.isfile(graph_path):
      graph_properties = torch.load(graph_path)
      return torch_geometric.data.Data(**graph_properties)
    else:
      raise ValueError(f'{graph_path} does not exist.')

def pre_construct_all_graphs_for_split(
    df,
    #
    data_path,
    #
    classification,
    #
    chunk_size,
    left_stride,
    right_stride,
    #
    surrogate,
    embedding_pooling,
    window_size,
    co_occurrence_pooling,
    #
    plm,
    tokenizer,
    maximum_chunk_size,
    #
    device
  ):
  for idx in range(df.shape[0]):
    graph_path = os.path.join(data_path, f'{idx}.pt')
    row = df.iloc[idx]
    document_level_graph = construct.construct_PyG_graph_from_sliding_windows(
      text = row['text'],
      label = row['label'],
      index = idx,
      classification = classification,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      surrogate = surrogate,
      embedding_pooling = embedding_pooling,
      window_size = window_size,
      co_occurrence_pooling = co_occurrence_pooling,
      #
      plm = plm,
      tokenizer = tokenizer,
      maximum_chunk_size = maximum_chunk_size,
      #
      device = device
    )
    # Store in disk
    torch.save(document_level_graph, graph_path)

    # print(f'Allocated: {torch.cuda.memory_allocated() / 1024**3:.4f} GB, Reserved:  {torch.cuda.memory_reserved() / 1024**3:.4f} GB', flush = True)

    # Clear memory
    del document_level_graph
    gc.collect()
    torch.cuda.empty_cache()
