import torch
import unicodedata

def get_logits_attentions_and_embeddings(model, input_identifiers, attention_mask):
  model.eval()
  model.zero_grad()
  outputs = model(
    input_ids = input_identifiers,
    attention_mask = attention_mask
  )
  embeddings = outputs['hidden_states'][-1][0].detach()
  logits = outputs.logits
  # Equation 3: Obtain raw attentions 
  attentions = outputs.attentions
  # Clear memory
  del outputs
  
  return logits, attentions, embeddings

# Inspired by https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb#scrollTo=TtqMdXdTEKAP
# and https://github.com/hila-chefer/Transformer-Explainability/blob/main/BERT_explainability/modules/BERT/BERT.py
def bert_generic_attention_explainability(
    input_identifiers,
    logits,
    attentions,
    label_index,
    device
  ):
  # Obtain attention gradients
  target_score = logits[:, label_index].sum()
  gradients = torch.autograd.grad(outputs = target_score, inputs = attentions, retain_graph = False) # True
  
  attentions = [x.detach() for x in attentions]
  gradients = [x.detach() for x in gradients]
  
  # Equation 1: Relevance initialization using identity matrix
  identity = torch.eye(input_identifiers.size(1), device = device)
  relevance_matrix = identity.detach().clone()
  # Iterate over all model layers
  for attention_matrix_with_batch, gradient_matrix_with_batch in zip(attentions, gradients):    
    attention_matrix = attention_matrix_with_batch[0]
    gradient_matrix = gradient_matrix_with_batch[0]
    # Equation 5: Hadamard product and removal of negative contributions
    contributions = torch.clamp(attention_matrix * gradient_matrix, min = 0.0)
    # Equation 5: Head-level averaging
    A_tilde = contributions.mean(dim = 0)
    # Equation 6: Relevance update
    relevance_matrix += torch.matmul(A_tilde, relevance_matrix)
    # Equation 9: Row normalization
    # relevance_matrix = relevance_matrix / (relevance_matrix.sum(dim = 1, keepdim = True) + 1e-10) 
    # Clear memory
    del contributions
    del A_tilde
  # Equation 9: Row normalization
  relevance_matrix_hat = relevance_matrix - identity
  relevance_matrix_tilde = relevance_matrix_hat / (relevance_matrix_hat.sum(dim = 1, keepdim = True) + 1e-10) + identity
  # Clear memory
  del target_score
  del gradients
  del identity
  del relevance_matrix
  del relevance_matrix_hat
  
  return relevance_matrix_tilde

def prune_using_importance_scores(input_identifiers, tokens, special_token_masking, importance, token_threshold, edge_threshold, drop_first_level_edges_ablation, device):  
  # Importance token-level importance with regards to [CLS]
  CLS_importance = importance[0]
  token_indices = torch.arange(input_identifiers.size(0)).to(device)
  # Drop special tokens
  CLS_importance_without_special_tokens = CLS_importance[~special_token_masking]
  importance_without_special_tokens = importance[~special_token_masking][:, ~special_token_masking]
  tokens_without_special_tokens = [x for i, x in enumerate(tokens) if not special_token_masking[i]]
  token_indices_without_special_tokens = token_indices[~special_token_masking]
  # input_identifiers_without_special_tokens = input_identifiers[~special_token_masking]
  # Drop punctuation tokens
  punctuation_mask = torch.tensor([all(unicodedata.category(c).startswith('P') for c in x) for x in tokens_without_special_tokens], device = device)
  CLS_importance_without_punctuation_tokens = CLS_importance_without_special_tokens[~punctuation_mask]
  importance_without_punctuation_tokens = importance_without_special_tokens[~punctuation_mask][:, ~punctuation_mask]
  tokens_without_punctuation_tokens = [x for i, x in enumerate(tokens_without_special_tokens) if not punctuation_mask[i]]
  token_indices_without_punctuation_tokens = token_indices_without_special_tokens[~punctuation_mask]
  # input_identifiers_without_punctuation_tokens = input_identifiers_without_special_tokens[~punctuation_mask]
  # Prune all tokens whose CLS-importance is below the Q-quantile
  token_cutting_point = torch.quantile(CLS_importance_without_punctuation_tokens, token_threshold).item()
  token_importance_mask = CLS_importance_without_punctuation_tokens >= token_cutting_point
  CLS_importance_without_unimportant_tokens = CLS_importance_without_punctuation_tokens[token_importance_mask]
  importance_without_unimportant_tokens = importance_without_punctuation_tokens[token_importance_mask][:, token_importance_mask]
  tokens_without_unimportant_tokens = [x for i, x in enumerate(tokens_without_punctuation_tokens) if token_importance_mask[i]]
  token_indices_without_unimportant_tokens = token_indices_without_punctuation_tokens[token_importance_mask]
  # input_identifiers_without_unimportant_tokens = input_identifiers_without_punctuation_tokens[token_importance_mask]
  # Prune edges by dropping all whose importance is below the Q-quantile
  off_diagonal_mask = ~torch.eye(importance_without_unimportant_tokens.size(0), dtype = torch.bool, device = device)
  if importance_without_unimportant_tokens[off_diagonal_mask].size(0) > 0:
    edge_cutting_point = torch.quantile(importance_without_unimportant_tokens[off_diagonal_mask], edge_threshold)
  else:
    edge_cutting_point = 0.0 # To avoid errors when importance matrix is empty
  if drop_first_level_edges_ablation:
    edge_cutting_point += 0.1 # To ensure that no off diagonal edge is kept
  edges_to_keep_mask = (importance_without_unimportant_tokens >= edge_cutting_point) & off_diagonal_mask
  importance_without_unimportant_relationships = importance_without_unimportant_tokens.clone()
  importance_without_unimportant_relationships[~edges_to_keep_mask & off_diagonal_mask] = 0.0
  # Clear memory
  del CLS_importance
  del token_indices
  del CLS_importance_without_special_tokens
  del importance_without_special_tokens
  del tokens_without_special_tokens
  del token_indices_without_special_tokens
  del punctuation_mask
  del CLS_importance_without_punctuation_tokens
  del importance_without_punctuation_tokens
  del tokens_without_punctuation_tokens
  del token_indices_without_punctuation_tokens
  del token_importance_mask
  del importance_without_unimportant_tokens
  del off_diagonal_mask
  del edges_to_keep_mask

  return CLS_importance_without_unimportant_tokens, importance_without_unimportant_relationships, tokens_without_unimportant_tokens, token_indices_without_unimportant_tokens

def construct_graph_from_importance_scores(
    label_indices,
    model,
    input_identifiers,
    attention_mask,
    special_token_masking,
    tokens,
    token_threshold,
    edge_threshold,
    drop_first_level_edges_ablation,
    unit_weight_edges_ablation,
    bi_directional_edges_to_second_and_third_level_nodes_ablation,
    chunk_identifier,
    device
  ):
  sub_graphs = list()
  #
  for label_index in label_indices:
    logits, attentions, embeddings = get_logits_attentions_and_embeddings(model = model, input_identifiers = input_identifiers, attention_mask = attention_mask)
    # Compute the generic Chefer importance for the text
    importance = bert_generic_attention_explainability(
      input_identifiers = input_identifiers,
      logits = logits,
      attentions = attentions,
      label_index = label_index,
      device = device
    )
    node_level_importance, edge_level_importance, node_labels, node_identifiers = prune_using_importance_scores(
      input_identifiers = input_identifiers[0],
      tokens = tokens,
      special_token_masking = special_token_masking,
      importance = importance,
      token_threshold = token_threshold,
      edge_threshold = edge_threshold,
      drop_first_level_edges_ablation = drop_first_level_edges_ablation,
      device = device
    )
    # Filter token embeddings
    node_embeddings = embeddings[node_identifiers].detach().cpu()
    node_identifiers = node_identifiers.detach().cpu()
    # Remove the extra unit weight given to every self-loop
    edge_level_importance = edge_level_importance - torch.eye(edge_level_importance.size(0), device = device)
    
    PyG_edges = edge_level_importance.detach().cpu().reshape(-1)
    important_edge_mask = PyG_edges != 0.0
    edge_index = torch.cartesian_prod(node_identifiers, node_identifiers).t().contiguous()[:, important_edge_mask].detach().cpu()
    edge_attr = PyG_edges[important_edge_mask].detach().cpu()
    node_level_importance = node_level_importance.detach().cpu()
    
    second_level_edges = torch.cartesian_prod(node_identifiers, torch.tensor([-chunk_identifier])).t().contiguous()
    if bi_directional_edges_to_second_and_third_level_nodes_ablation:
      second_level_edges = torch.cat((second_level_edges, second_level_edges.flip(dims = (0, ))), dim = 1)
      node_level_importance = torch.cat((node_level_importance, node_level_importance), dim = 0)
    
    sub_graphs.append({
      'edge_index' : torch.cat((edge_index, second_level_edges), dim = 1),
      'edge_attr' : torch.cat((edge_attr, node_level_importance), dim = 0),
      'node_index' : node_identifiers,
      'node_label' : node_labels,
      'node_attr' : node_embeddings
    })

    # Clear memory
    del importance
    del node_level_importance
    del edge_level_importance
    del node_labels
    del node_identifiers
    del node_embeddings
    del PyG_edges
    del important_edge_mask
    del edge_index
    del edge_attr
    del second_level_edges

  # Clear memory
  del logits
  del attentions
  del embeddings

  document_level_graph = {
    'nodes' : {},
    'edges' : {}
  }

  for label_index, sub_graph in enumerate(sub_graphs):
    edge_index = sub_graph['edge_index']
    edge_attr = sub_graph['edge_attr']
    node_index = sub_graph['node_index']
    node_label = sub_graph['node_label']
    node_attr = sub_graph['node_attr']

    # Second-level node
    document_level_graph['nodes'][(-chunk_identifier, f'[T-{chunk_identifier}]')] = torch.ones(model.config.hidden_size).tolist()
    for node, label, embedding in zip(node_index.tolist(), node_label, node_attr.tolist()):
      key = (node, label)
      if key not in document_level_graph['nodes']:
        document_level_graph['nodes'][key] = embedding
    
    document_level_graph['edges'][(-chunk_identifier, -chunk_identifier)] = torch.ones(len(label_indices))
    for (source, target), weight in zip(edge_index.t().tolist(), edge_attr.tolist()):
      key = (source, target)
      if key not in document_level_graph['edges']:
        document_level_graph['edges'][key] = torch.zeros(len(label_indices))
      document_level_graph['edges'][key][label_index] = weight if not unit_weight_edges_ablation else 1.0
    
    # Clear memory
    del edge_index
    del edge_attr
    del node_index
    del node_label
    del node_attr
  
  # Clear memory
  del sub_graphs

  return torch.tensor([x for x, _ in list(document_level_graph['nodes'].keys())]), \
    [y for _, y in list(document_level_graph['nodes'].keys())], \
    torch.tensor(list(document_level_graph['nodes'].values())), \
    torch.tensor(list(document_level_graph['edges'].keys()), dtype = torch.long).t().contiguous(), \
    torch.tensor([value.tolist() for value in document_level_graph['edges'].values()])

def hierarchically_aggregate_chunk_level_subgraphs(
    chunks,
    embedding_dimension,
    drop_second_level_nodes_ablation,
    drop_second_and_third_level_nodes_ablation
  ):

  chunk_count = len(chunks) + 1
  last_used_index = chunk_count
  for chunk in chunks:
    new_indices = dict()
    for node in chunk['node_identifiers'].tolist():
      if node < 0:
        new_indices[node] = -node
      elif node not in new_indices:
        new_indices[node] = last_used_index
        last_used_index += 1
    updated_identifiers = chunk['node_identifiers'].clone()
    updated_edge_identifiers = chunk['edge_identifiers'].clone()
    for k, v in new_indices.items():
      updated_identifiers[chunk['node_identifiers'] == k] = v
      updated_edge_identifiers[chunk['edge_identifiers'] == k] = v
    chunk['node_identifiers'] = updated_identifiers
    chunk['edge_identifiers'] = updated_edge_identifiers
    # Clear memory
    del updated_identifiers
    del updated_edge_identifiers

  document_level_graph = {
    'node_identifiers' : torch.cat([torch.tensor([0], dtype = torch.long)] + [chunk['node_identifiers'] for chunk in chunks], dim = 0),
    'node_labels' : [x for x_ in [['[D]']] + [chunk['node_labels'] for chunk in chunks] for x in x_],
    'node_attributes' : torch.cat([torch.ones(1, embedding_dimension)] + [chunk['node_attributes'] for chunk in chunks], dim = 0),
    'edge_identifiers' : torch.cat([torch.tensor([(i, 0) for i in range(chunk_count)], dtype = torch.long).t().contiguous()] + [chunk['edge_identifiers'] for chunk in chunks], dim = 1),
    'edge_weights' : torch.cat([torch.tensor([torch.ones(len(chunks[0]['edge_weights'][0])).tolist() for _ in range(chunk_count)])] + [chunk['edge_weights'] for chunk in chunks], dim = 0),
  }

  # Clear memory
  del chunks

  sorted_document_level_node_identifiers, sort_indices_for_document_level_node_identifiers = torch.sort(document_level_graph['node_identifiers'])
  document_level_graph['node_identifiers'] = sorted_document_level_node_identifiers
  document_level_graph['node_labels'] = [document_level_graph['node_labels'][i] for i in sorted_document_level_node_identifiers.tolist()]
  document_level_graph['node_attributes'] = document_level_graph['node_attributes'][sort_indices_for_document_level_node_identifiers]

  if drop_second_level_nodes_ablation:
    # Remove second-level nodes ([1, ..., chunk_count])
    second_level_nodes_mask = (document_level_graph['node_identifiers'] >= 1) & (document_level_graph['node_identifiers'] < chunk_count)
    document_level_graph['node_identifiers'] = document_level_graph['node_identifiers'][~second_level_nodes_mask]
    document_level_graph['node_labels'] = [x for i, x in enumerate(document_level_graph['node_labels']) if not second_level_nodes_mask[i]]
    document_level_graph['node_attributes'] = document_level_graph['node_attributes'][~second_level_nodes_mask]
    # Remove edges from [1, ..., chunk_count] to 0 and [1, ..., chunk_count] self-loops
    source_nodes, target_nodes = document_level_graph['edge_identifiers']
    third_level_edges_mask = ((source_nodes >= 1) & (source_nodes < chunk_count) & (target_nodes == 0)) | ((source_nodes >= 1) & (source_nodes < chunk_count) & (target_nodes >= 1) & (target_nodes < chunk_count))
    document_level_graph['edge_identifiers'] = document_level_graph['edge_identifiers'][:, ~third_level_edges_mask]
    document_level_graph['edge_weights'] = document_level_graph['edge_weights'][~third_level_edges_mask]
    # Replace node identifiers in edges that contain [1, ..., chunk_count] with 0 (third-level node)
    _, target_nodes = document_level_graph['edge_identifiers']
    second_level_edges_mask = (target_nodes >= 1) & (target_nodes < chunk_count)
    document_level_graph['edge_identifiers'][1][second_level_edges_mask] = 0
    # Update node and edge identifiers 
    document_level_graph['node_identifiers'][document_level_graph['node_identifiers'] > 0] -= (chunk_count - 1)
    document_level_graph['edge_identifiers'][document_level_graph['edge_identifiers'] > 0] -= (chunk_count - 1)
  elif drop_second_and_third_level_nodes_ablation:
    # Remove second-level nodes ([1, ..., chunk_count]) and third-level node
    second_and_third_level_nodes_mask = (document_level_graph['node_identifiers'] >= 0) & (document_level_graph['node_identifiers'] < chunk_count)
    document_level_graph['node_identifiers'] = document_level_graph['node_identifiers'][~second_and_third_level_nodes_mask]
    document_level_graph['node_labels'] = [x for i, x in enumerate(document_level_graph['node_labels']) if not second_and_third_level_nodes_mask[i]]
    document_level_graph['node_attributes'] = document_level_graph['node_attributes'][~second_and_third_level_nodes_mask]
    # Remove edges from/to second-level nodes ([1, ..., chunk_count]) and third-level node (0)
    source_nodes, target_nodes = document_level_graph['edge_identifiers']
    second_and_third_level_edges_mask = (source_nodes < chunk_count) | (target_nodes < chunk_count)
    document_level_graph['edge_identifiers'] = document_level_graph['edge_identifiers'][:, ~second_and_third_level_edges_mask]
    document_level_graph['edge_weights'] = document_level_graph['edge_weights'][~second_and_third_level_edges_mask]
    # Update node and edge identifiers 
    document_level_graph['node_identifiers'][document_level_graph['node_identifiers'] > 0] -= chunk_count
    document_level_graph['edge_identifiers'][document_level_graph['edge_identifiers'] > 0] -= chunk_count
  
  # Clear memory
  del sorted_document_level_node_identifiers
  del sort_indices_for_document_level_node_identifiers
  
  return document_level_graph
