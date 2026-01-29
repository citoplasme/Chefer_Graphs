import typing
import torch
import chefer_importance

def construct_PyG_graph_from_Chefer_importance(
    text : str,
    label : int | float,
    #split : str,
    index : int,
    #
    label_indices : typing.List[int],
    #
    chunk_size : int,
    left_stride : int,
    right_stride : int,
    #
    token_threshold : float,
    edge_threshold : float,
    #
    plm,
    tokenizer,
    maximum_chunk_size : int,
    #
    drop_first_level_edges_ablation : bool,
    unit_weight_edges_ablation : bool,
    drop_second_level_nodes_ablation : bool,
    drop_second_and_third_level_nodes_ablation : bool,
    #
    device,
  ):
  
  if maximum_chunk_size < chunk_size:
    raise ValueError('The selected chunk size is not accepted by the pre-trained model.')

  if left_stride + right_stride >= chunk_size:
    raise ValueError('The selected strides are greater or equal to the total chunk size.')
  
  if token_threshold < 0 or token_threshold > 1:
    raise ValueError('The token-level threshold must be between 0 and 1.')

  if edge_threshold < 0 or edge_threshold > 1:
    raise ValueError('The edge-level threshold must be between 0 and 1.')
  
  encoding = tokenizer(
    text,
    return_tensors = 'pt',
    padding = False,
    truncation = False
  )
  input_identifiers = encoding['input_ids'].to(device)
  attention_mask = encoding['attention_mask'].to(device)
  special_token_masking = torch.tensor(tokenizer.get_special_tokens_mask(input_identifiers[0], already_has_special_tokens = True), dtype = torch.bool).to(device)
  tokens = tokenizer.convert_ids_to_tokens(input_identifiers[0])

  document_length = input_identifiers.size(1)

  plm.eval()

  c = chunk_size - left_stride - right_stride

  chunks = list()

  # Single chunk when document fits inside the context window
  if document_length <= maximum_chunk_size:
    left_stride = 0
    right_stride = 0
    c = document_length  

  # Compute the information (attention coefficients and embeddings) for each chunk
  s = 1
  identifier = 1
  while s < document_length - 1:
    l = max(1, s - left_stride)
    end_index = s + c - 2 if l > 1 else s + c + left_stride - 2
    r = min(document_length - 1, end_index + right_stride)
    
    # Chunk the input sequences and add special tokens to start and end of each chunk
    chunk_input_identifiers = torch.cat((input_identifiers[:, 0].reshape(1, 1), input_identifiers[:, l : r], input_identifiers[:, document_length - 1].reshape(1, 1)), dim = 1)
    chunk_attention_masking = torch.cat((attention_mask[:, 0].reshape(1, 1), attention_mask[:, l : r], attention_mask[:, document_length - 1].reshape(1, 1)), dim = 1)
    chunk_special_token_masking = torch.cat((special_token_masking[0].reshape(1), special_token_masking[l : r], special_token_masking[document_length - 1].reshape(1)), dim = 0)
    chunk_tokens = [tokens[0]] + tokens[l : r] + [tokens[document_length - 1]]

    node_indices, node_labels, node_attrs, edge_indices, edge_attrs = chefer_importance.construct_graph_from_importance_scores(
      label_indices = label_indices,
      model = plm,
      input_identifiers = chunk_input_identifiers,
      attention_mask = chunk_attention_masking,
      special_token_masking = chunk_special_token_masking,
      tokens = chunk_tokens,
      token_threshold = token_threshold,
      edge_threshold = edge_threshold,
      drop_first_level_edges_ablation = drop_first_level_edges_ablation,
      unit_weight_edges_ablation = unit_weight_edges_ablation,
      chunk_identifier = identifier,
      device = device
    )
    chunks.append({
      'node_identifiers' : node_indices,
      'node_attributes' : node_attrs,
      'node_labels' : node_labels,
      'edge_identifiers' : edge_indices,
      'edge_weights' : edge_attrs
    })

    identifier += 1

    s = end_index - 1

    if r == document_length - 1:
      break
  
  document_level_graph = chefer_importance.hierarchically_aggregate_chunk_level_subgraphs(
    chunks = chunks,
    embedding_dimension = plm.config.hidden_size,
    drop_second_level_nodes_ablation = drop_second_level_nodes_ablation,
    drop_second_and_third_level_nodes_ablation = drop_second_and_third_level_nodes_ablation
  )

  document_level_graph_PyG = {
    'tokens' : document_level_graph['node_labels'],
    'x' : document_level_graph['node_attributes'].to('cpu'),
    'y' : torch.tensor([label], dtype = torch.long).to('cpu'),
    'edge_index' : document_level_graph['edge_identifiers'].to('cpu'),
    'edge_attr' : document_level_graph['edge_weights'].to('cpu'),
    'identifier' : torch.tensor([index], dtype = torch.long).to('cpu')
  }
  
  return document_level_graph_PyG
