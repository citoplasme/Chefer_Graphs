import torch
# import torch_geometric
import gc

# Adapted from https://github.com/jessevig/bertviz/blob/master/bertviz/head_view.py
def format_attention(attention_coefficients):
  squeezed = []
  for layer_attention in attention_coefficients:
    if len(layer_attention.shape) != 4: # 1, #heads, sequence length, sequence length
      raise ValueError('The attention tensor does not have the correct number of dimensions: 1, number of attention heads, sequence length, sequence length.')
    layer_attention = layer_attention.squeeze(0)
    squeezed.append(layer_attention)
  return torch.stack(squeezed) # #layers, #heads, sequence length, sequence length

def get_pooling_operation_from_string(pooling_operation):
  OPERATIONS = {
    'mean' : torch.mean,
    #'median' : torch.median,
    'max' : torch.max,
    'min' : torch.min,
    'sum' : torch.sum
  }
  return OPERATIONS[pooling_operation]

def embedding_scatter_reduce(index, x, pooling_operation, device):
  
  unique, inverse_indices = torch.unique(index, dim = 0, return_inverse = True)
  
  if pooling_operation == 'sum':
    agg = torch.zeros(unique.size(0), x.size(1), dtype = x.dtype).to(device)
    agg.scatter_reduce_(dim = 0, index = inverse_indices.unsqueeze(1).expand(-1, x.size(1)), src = x, reduce = 'sum') # sum, prod, mean, amax, amin
  
  elif pooling_operation == 'mean':
    agg_sum = torch.zeros(unique.size(0), x.size(1), dtype = x.dtype).to(device)
    agg_count = torch.zeros(unique.size(0), x.size(1), dtype = x.dtype).to(device)

    agg_sum.scatter_add_(dim = 0, index = inverse_indices.unsqueeze(1).expand(-1, x.size(1)), src = x)
    agg_count.scatter_add_(dim = 0, index = inverse_indices.unsqueeze(1).expand(-1, x.size(1)), src = torch.ones_like(x).to(device))

    agg = agg_sum / agg_count
  
  elif pooling_operation == 'min':
    agg = torch.full((unique.size(0), x.size(1)), float('inf'), dtype = x.dtype).to(device)
    agg.scatter_reduce_(dim = 0, index = inverse_indices.unsqueeze(1).expand(-1, x.size(1)), src = x, reduce = 'amin')
  
  elif pooling_operation == 'max':
    agg = torch.full((unique.size(0), x.size(1)), float('-inf'), dtype = x.dtype).to(device)
    agg.scatter_reduce_(dim = 0, index = inverse_indices.unsqueeze(1).expand(-1, x.size(1)), src = x, reduce = 'amax')
  
  else:
    raise ValueError('Invalid pooling operation.')
  
  return unique, agg

def remove_isolated_nodes(x, edge_index, device):
  connected_nodes = torch.unique(edge_index).to(device)

  mask = torch.zeros(x.size(0), dtype = torch.bool).to(device)
  mask[connected_nodes] = True
  x_without_isolated_nodes = x[mask]

  # Re-index edge_index
  new_indices = torch.zeros(x.size(0), dtype = torch.long).to(device) - 1 # -1 = isolated node
  new_indices[connected_nodes] = torch.arange(connected_nodes.size(0)).to(device)
  edge_index_without_isolated_nodes = new_indices[edge_index]

  return edge_index_without_isolated_nodes, x_without_isolated_nodes

def construct_PyG_graph_from_raw_attentions(
    text : str,
    label : int | float,
    #split : str,
    index : int,
    #
    chunk_size : int,
    left_stride : int,
    right_stride : int,
    #
    surrogate : bool,
    attention_pooling : str,
    embedding_pooling : str,
    threshold : float,
    aggregation_level : int,
    #
    plm,
    tokenizer,
    attention_output_key : str,
    maximum_chunk_size : int,
    layers : int,
    heads : int,
    #
    device,
    classification
  ):
  
  if maximum_chunk_size < chunk_size:
    raise ValueError('The selected chunk size is not accepted by the pre-trained model.')

  if left_stride + right_stride >= chunk_size:
    raise ValueError('The selected strides are greater or equal to the total chunk size.')
  
  if attention_pooling not in ['mean', 'sum', 'max', 'min']:
    raise ValueError('The selected attention pooling operation is not valid. Please choose one of the following: mean, sum, max, min.')
  
  if embedding_pooling not in ['mean', 'sum', 'max', 'min']:
    raise ValueError('The selected attention pooling operation is not valid. Please choose one of the following: mean, sum, max, min.')
  
  if threshold < 0 or threshold > 1:
    raise ValueError('The threshold must be between 0 and 1.')
  
  if aggregation_level not in [0, 1, 2]:
    raise ValueError('The aggregation level must be 0 (no aggregation), 1 (layer-wise aggregation), or 2 (global aggregation).')
  
  input_identifiers = tokenizer.encode(text, return_tensors = 'pt').to(device)
  special_token_masking = torch.tensor(tokenizer.get_special_tokens_mask(input_identifiers[0], already_has_special_tokens = True), dtype = torch.bool).to(device)
  # tokens = tokenizer.convert_ids_to_tokens(input_identifiers[0])

  document_length = input_identifiers.size(1)

  plm.eval()

  c = chunk_size - left_stride - right_stride

  node_list = list()
  edge_list = list()

  # Single chunk when document fits inside the context window
  if document_length <= maximum_chunk_size:
    left_stride = 0
    right_stride = 0
    c = document_length  

  # Compute the information (attention coefficients and embeddings) for each chunk
  s = 1
  while s < document_length - 1:
    l = max(1, s - left_stride)
    end_index = s + c - 2 if l > 1 else s + c + left_stride - 2
    r = min(document_length - 1, end_index + right_stride)
    
    # Chunk the input sequences and add special tokens to start and end of each chunk
    chunk_input_identifiers = torch.cat((input_identifiers[:, 0].reshape(1, 1), input_identifiers[:, l : r], input_identifiers[:, document_length - 1].reshape(1, 1)), dim = 1)
    chunk_special_token_masking = torch.cat((special_token_masking[0].reshape(1), special_token_masking[l : r], special_token_masking[document_length - 1].reshape(1)), dim = 0)
    #chunk_tokens = [tokens[0]] + tokens[l : r] + [tokens[document_length - 1]]

    with torch.no_grad():
      chunk_outputs = plm(chunk_input_identifiers)
      chunk_attention = format_attention(chunk_outputs[attention_output_key])
      chunk_embeddings = chunk_outputs['hidden_states'][-1][0].detach()

    chunk_attention_T = torch.movedim(chunk_attention, source = (2, 3), destination = (0, 1)).reshape((chunk_input_identifiers.size(1), chunk_input_identifiers.size(1), layers * heads))

    if surrogate:
      node_indices = torch.arange(l - 1, r + 1).to(device)
    else:
      node_indices = chunk_input_identifiers[0]
    
    source_indices = node_indices.view(-1, 1, 1).expand(-1, chunk_input_identifiers.size(1), -1)
    target_indices = node_indices.view(1, -1, 1).expand(chunk_input_identifiers.size(1), -1, -1)

    edge_list.append(
      (
        source_indices[~chunk_special_token_masking][:, ~chunk_special_token_masking],
        target_indices[~chunk_special_token_masking][:, ~chunk_special_token_masking],
        chunk_attention_T[~chunk_special_token_masking][:, ~chunk_special_token_masking].reshape(-1, layers * heads)
      )
    )

    node_list.append(
      (
        node_indices[~chunk_special_token_masking],
        chunk_embeddings[~chunk_special_token_masking]
      )
    )

    s = end_index - 1

    if r == document_length - 1:
      break
  
  edge_index = torch.cat([torch.cat((x[0].reshape(-1, 1), x[1].reshape(-1, 1)), dim = 1) for x in edge_list], dim = 0)
  edge_attr = torch.cat([x[2] for x in edge_list], dim = 0)
  del edge_list
  
  node_index = torch.cat([x[0] for x in node_list], dim = 0)
  node_attr = torch.cat([x[1] for x in node_list], dim = 0)
  del node_list

  # Aggregation of attention coefficients across all chunks
  edge_index_unique, edge_attr_aggregated = embedding_scatter_reduce(edge_index, edge_attr, attention_pooling, device = device)
  del edge_index
  del edge_attr

  # Apply the requested level of aggregation to the edge embeddings
  if aggregation_level == 1: # Layer-wise
    edge_attr_aggregated = get_pooling_operation_from_string(attention_pooling)(edge_attr_aggregated.view(edge_attr_aggregated.size(0), layers, heads), dim = 2)
  elif aggregation_level == 2: # Global
    edge_attr_aggregated = get_pooling_operation_from_string(attention_pooling)(edge_attr_aggregated, dim = 1, keepdim = True)  
  if not torch.is_tensor(edge_attr_aggregated): # attention_pooling not in ['mean', 'sum']:
    edge_attr_aggregated = edge_attr_aggregated.values

  # Calculate pooled attention, i.e., a single value used to apply the threshold 
  aggregated_attention = get_pooling_operation_from_string(attention_pooling)(edge_attr_aggregated, dim = 1)
  if not torch.is_tensor(aggregated_attention): # attention_pooling not in ['mean', 'sum']:
    aggregated_attention = aggregated_attention.values

  # Threshold
  cutting_point = torch.quantile(aggregated_attention, threshold).item() # np.quantile(aggregated_attention, threshold)
  threshold_mask = aggregated_attention >= cutting_point
  edge_index_unique = edge_index_unique[threshold_mask]
  edge_attr_aggregated = edge_attr_aggregated[threshold_mask]

  # Aggregate embeddings across chunks
  node_index_unique, node_attr_aggregated = embedding_scatter_reduce(node_index, node_attr, embedding_pooling, device = device)
  del node_index
  del node_attr

  # Replace edge identifiers with PyG-compatible node identifiers
  edge_index_unique_PyG = torch.zeros_like(edge_index_unique).to(device)
  for idx, value in enumerate(node_index_unique): # Unique always returns sorted values
    edge_index_unique_PyG = torch.where(edge_index_unique == value, idx, edge_index_unique_PyG)

  # Drop isolated nodes
  edge_index_unique_PyG_without_isolated_nodes, node_attr_aggregated_without_isolated_nodes = remove_isolated_nodes(
    x = node_attr_aggregated,
    edge_index = edge_index_unique_PyG.t().contiguous(),
    device = device
  )
  
  # # Convert to PyG graph
  # document_graph_PyG = torch_geometric.data.Data(
  #   x = node_attr_aggregated_without_isolated_nodes,
  #   y = torch.tensor([label], dtype = torch.long),
  #   edge_index = edge_index_unique_PyG_without_isolated_nodes,
  #   edge_attr = edge_attr_aggregated,
  #   split = split,
  #   identifier = torch.tensor([index], dtype = torch.long)
  # ).to(device) #.to('cpu')
  document_graph_PyG = {
    'x' : node_attr_aggregated_without_isolated_nodes.to('cpu'),
    'y' : torch.tensor([label], dtype = torch.long if classification else torch.float).to('cpu'),
    'edge_index' : edge_index_unique_PyG_without_isolated_nodes.to('cpu'),
    'edge_attr' : edge_attr_aggregated.to('cpu'),
    'identifier' : torch.tensor([index], dtype = torch.long).to('cpu')
  }

  # Delete variables
  del edge_index_unique
  del edge_attr_aggregated
  del aggregated_attention
  del cutting_point
  del threshold_mask
  del node_index_unique
  del node_attr_aggregated
  del edge_index_unique_PyG
  del edge_index_unique_PyG_without_isolated_nodes
  del node_attr_aggregated_without_isolated_nodes
  
  gc.collect()
  torch.cuda.empty_cache()
  
  return document_graph_PyG