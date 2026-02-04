import torch
import torch_geometric

# ===========================================================================
# ============================= Model definition ============================
# ===========================================================================

# Based on https://mlabonne.github.io/blog/posts/2022-03-09-Graph_Attention_Network.html
class GATv2(torch.nn.Module):
  def __init__(
    self,
    node_feature_count : int,
    class_count : int,
    hidden_dimension : int = 64,
    edge_dimension : int = None,
    attention_heads : int = 8,
    number_of_hidden_layers : int = 0,
    dropout_rate : float = 0.6,
  ):
    super(GATv2, self).__init__()

    self.convs = torch.nn.ModuleList()
    self.convs.append(torch_geometric.nn.GATv2Conv(in_channels = node_feature_count, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False, concat = True if number_of_hidden_layers > 0 else False))

    self.norms = torch.nn.ModuleList()
    self.norms.append(torch_geometric.nn.norm.GraphNorm(in_channels = hidden_dimension * attention_heads if number_of_hidden_layers > 0 else hidden_dimension))

    for i in range(number_of_hidden_layers):
      self.convs.append(torch_geometric.nn.GATv2Conv(in_channels = hidden_dimension * attention_heads, out_channels = hidden_dimension, edge_dim = edge_dimension, heads = attention_heads, add_self_loops = False, concat = True if i < number_of_hidden_layers -1 else False))
      self.norms.append(torch_geometric.nn.norm.GraphNorm(in_channels = hidden_dimension * attention_heads if i < number_of_hidden_layers -1 else hidden_dimension))

    self.linear = torch.nn.Linear(hidden_dimension, class_count)
    self.dropout_rate = dropout_rate

  def forward(self, x, edge_index, edge_attr, batch):
    # 1. Obtain node embeddings
    for conv, norm in zip(self.convs, self.norms):
      x = conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
      x = torch.nn.functional.leaky_relu(x)
      x = torch.nn.functional.dropout(x, p = self.dropout_rate, training = self.training)
      x = norm(x)
    
    # 2. Readout layer
    batch_size = batch.max().item() + 1
    third_level_nodes_indices = torch.zeros(batch_size, dtype = torch.long, device = x.device)
    for i in range(batch_size):
      third_level_nodes_indices[i] = (batch == i).nonzero(as_tuple = False)[0]
    x = x[third_level_nodes_indices]

    # 3. Apply a final classifier
    return self.linear(x)
