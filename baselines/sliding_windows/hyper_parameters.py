HYPER_PARAMETERS = {
  'batch_size' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 32
  },
  'window_size' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [2, 3, 4, 5]
  },
  'attention_heads' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [1, 2, 4, 6]
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
    'original_search_space' : [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
  },
  'weight_decay' : {
    'type' : 'float',
    'log' : True,
    'fixed' : False,
    'original_search_space' : [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
  },
  'epochs' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 30
  },
  'balanced_loss' : {
    'type' : 'categorical',
    'fixed' : True,
    'value' : True
  },
  'early_stopping_patience' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 5
  },
  'early_stopping_start_epoch' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 10
  },
  'linear_warmup_epochs' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 5
  },
  'linear_warmup_start_factor' : {
    'type' : 'float',
    'log' : False,
    'fixed' : True,
    'value' : 0.1
  },
  'cosine_annealing_eta_min' : {
    'type' : 'float',
    'log' : True,
    'fixed' : True,
    'value' : 1e-6
  },
  'cosine_annealing_T_0' : {
    'type' : 'int',
    'fixed' : True,
    'value' : 5
  },
  'cosine_annealing_T_mult' : {
    'type' : 'int',
    'fixed' : False,
    'original_search_space' : [1, 2]
  },
  'label_smoothing' : {
    'type' : 'float',
    'log' : False,
    'fixed' : False,
    'original_search_space' : [0.0, 0.05, 0.1, 0.15]
  },
  'gradient_clipping' : {
    'type' : 'float',
    'log' : False,
    'fixed' : False,
    'original_search_space' : [0.5, 1.0, 1.5, 2.0]
  }
}
