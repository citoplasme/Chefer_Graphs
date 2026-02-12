HYPER_PARAMETERS = {
  'batch_size' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [16, 32, 64, 128]
  },
  'edge_threshold' : {
    'type' : 'float',
    'log' : False,
    'fixed' : False,
    'original_search_space' : [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
  },
  'node_threshold' : {
    'type' : 'float',
    'log' : False,
    'fixed' : False,
    'original_search_space' : [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
  },
  'attention_heads' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [1, 2, 4, 6, 8]
  },
  'hidden_dimension' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [32, 64, 128]
  },
  'number_of_hidden_layers' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [0, 1, 2]
  },
  'dropout_rate' : {
    'type' : 'float',
    'log' : False,
    'fixed' : False,
    'original_search_space' : [0.1, 0.2, 0.3, 0.4]
  },
  'learning_rate' : {
    'type' : 'float',
    'log' : True,
    'fixed' : False,
    'original_search_space' : [1e-5, 1e-4, 3e-4, 1e-3]
  },
  'weight_decay' : {
    'type' : 'float',
    'log' : True,
    'fixed' : False,
    'original_search_space' : [1e-6, 1e-5, 1e-4, 1e-3]
  },
  'epochs' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 150
  },
  # 'balanced_loss' : {
  #   'type' : 'categorical',
  #   'fixed' : True,
  #   'value' : True
  # },
  'early_stopping_patience' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 10
  },
  'early_stopping_start_epoch' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 10
  },
  'linear_warmup_step_ratio' : {
    'type' : 'float',
    'log' : False,
    'fixed' : True,
    'value' : 0.1
  },
  'linear_warmup_start_factor' : {
    'type' : 'float',
    'log' : False,
    'fixed' : True,
    'value' : 0.1
  },
  'linear_decay_end_factor' : {
    'type' : 'float',
    'log' : True,
    'fixed' : True,
    'value' : 1e-4
  },
  # 'label_smoothing' : {
  #   'type' : 'float',
  #   'log' : False,
  #   'fixed' : False,
  #   'original_search_space' : [0.0, 0.05, 0.1, 0.15]
  # },
  # 'gradient_clipping' : {
  #   'type' : 'float',
  #   'log' : False,
  #   'fixed' : False,
  #   'original_search_space' : [0.5, 1.0, 1.5, 2.0]
  # }
}
