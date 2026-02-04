import os
import torch
import torch_geometric
import gc

import sys
sys.path.append(os.path.join('..', 'graph_construction'))
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
    label_indices,
    #
    chunk_size,
    left_stride,
    right_stride,
    #
    token_threshold,
    edge_threshold,
    #
    plm,
    tokenizer,
    maximum_chunk_size,
    #
    drop_first_level_edges_ablation,
    unit_weight_edges_ablation,
    drop_second_level_nodes_ablation,
    drop_second_and_third_level_nodes_ablation,
    bi_directional_edges_to_second_and_third_level_nodes_ablation,
    #
    device
  ):
  for idx in range(df.shape[0]):
    graph_path = os.path.join(data_path, f'{idx}.pt')
    row = df.iloc[idx]
    document_level_graph = construct.construct_PyG_graph_from_Chefer_importance(
      text = row['text'],
      label = row['label'],
      index = idx,
      #
      label_indices = label_indices,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      token_threshold = token_threshold,
      edge_threshold = edge_threshold,
      #
      plm = plm,
      tokenizer = tokenizer,
      maximum_chunk_size = maximum_chunk_size,
      #
      drop_first_level_edges_ablation = drop_first_level_edges_ablation,
      unit_weight_edges_ablation = unit_weight_edges_ablation,
      drop_second_level_nodes_ablation = drop_second_level_nodes_ablation,
      drop_second_and_third_level_nodes_ablation = drop_second_and_third_level_nodes_ablation,
      bi_directional_edges_to_second_and_third_level_nodes_ablation = bi_directional_edges_to_second_and_third_level_nodes_ablation,
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
