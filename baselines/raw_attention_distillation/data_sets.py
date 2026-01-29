import os
import torch
import torch_geometric

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
      del graph_properties['tokens']
      return torch_geometric.data.Data(**graph_properties)
    else:
      raise ValueError(f'{graph_path} does not exist.')

def pre_construct_all_graphs_for_split(
    df,
    #
    data_path,
    #
    chunk_size,
    left_stride,
    right_stride,
    #
    surrogate,
    attention_pooling,
    embedding_pooling,
    threshold,
    aggregation_level,
    #
    plm,
    tokenizer,
    attention_output_key,
    embedding_output_key,
    maximum_chunk_size,
    layers,
    heads,
    #
    device,
    classification
  ):
  for idx in range(df.shape[0]):
    graph_path = os.path.join(data_path, f'{idx}.pt')
    row = df.iloc[idx]
    document_level_graph = construct.construct_PyG_graph_from_raw_attentions(
      text = row['text'],
      label = row['label'],
      index = idx,
      #
      classification = classification,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      surrogate = surrogate,
      attention_pooling = attention_pooling,
      embedding_pooling = embedding_pooling,
      threshold = threshold,
      aggregation_level = aggregation_level,
      #
      plm = plm,
      tokenizer = tokenizer,
      attention_output_key = attention_output_key,
      embedding_output_key = embedding_output_key,
      maximum_chunk_size = maximum_chunk_size,
      layers = layers,
      heads = heads,
      #
      device = device
    )
    # Store in disk
    torch.save(document_level_graph, graph_path)
