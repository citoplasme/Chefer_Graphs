import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch_geometric
import numpy as np
import pandas as pd
import optuna
import random
import gc
import time
import sklearn.metrics
import sklearn.utils
import argparse
import transformers
import statistics
import shutil

import hyper_parameters
import data_sets
import models

import warnings
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

st = time.time()

SEED = 42

TOP_N = 1#3
TEST_RUNS_AROUND_BASE_SEED = (0, 0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STORAGE_PATH = '../outputs/ablations/'
CACHE_PATH = os.path.join('.', 'models')
os.makedirs(CACHE_PATH, exist_ok = True)

SCRATCH_PATH = '/scratch/jpimen/XAI_Graph_Construction/cache/'
os.makedirs(SCRATCH_PATH, exist_ok = True)

CHECKPOINT_PATH = f'/scratch/jpimen/checkpoints/{os.environ["SLURM_JOB_ID"]}/'
os.makedirs(CHECKPOINT_PATH, exist_ok = True)

# ===========================================================================
# ============================= Auxiliary methods ===========================
# ===========================================================================

# From https://docs.pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

# ===========================================================================
# ============================== Model training =============================
# ===========================================================================

def model_training(
    training_batches,
    validation_batches,
    #
    attention_heads,
    hidden_dimension,
    number_of_hidden_layers,
    dropout_rate,
    #
    learning_rate,
    weight_decay,
    balanced_loss,
    epochs,
    #
    early_stopping_patience,
    early_stopping_start_epoch,
    #
    linear_warmup_step_ratio = None,
    linear_warmup_start_factor = None,
    linear_decay_end_factor = None,
    #
    label_smoothing = 0.0,
    gradient_clipping = None
  ):

  document_level_training_labels = pd.read_csv(os.path.join('..', 'data', 'with_validation_splits', DATASET, 'train.csv'))['label'].values
    
  model = models.GATv2(
    node_feature_count = EMBEDDING_DIMENSION,
    class_count = np.unique(document_level_training_labels).size,
    attention_heads = attention_heads,
    edge_dimension = np.unique(document_level_training_labels).size,
    hidden_dimension = hidden_dimension,
    number_of_hidden_layers = number_of_hidden_layers,
    dropout_rate = dropout_rate,
    global_pooling = 'mean' if (ABLATION == 4) else None
  ).to(DEVICE)
  
  decay_params = list()
  no_decay_params = list()
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if param.ndim == 1 or 'norm' in name.lower():
      no_decay_params.append(param)
    else:
      decay_params.append(param)  
  # optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = weight_decay)
  optimizer = torch.optim.AdamW(
    [
      {'params' : decay_params, 'weight_decay' : weight_decay},
      {'params' : no_decay_params, 'weight_decay' : 0.0},
    ],
    lr = learning_rate,
    betas = (0.9, 0.999),
    eps = 1e-08,
  )
  
  total_steps = epochs * len(training_batches)
  linear_warmup_steps = max(1, int(linear_warmup_step_ratio * total_steps))

  linear_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor = linear_warmup_start_factor,
    end_factor = 1.0,
    total_iters = linear_warmup_steps
  )
  linear_decay_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor = 1.0,
    end_factor = linear_decay_end_factor,
    total_iters = max(1, total_steps - linear_warmup_steps)
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers = [linear_warmup_scheduler, linear_decay_scheduler], milestones = [linear_warmup_steps])
  
  # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
  if balanced_loss:
    CLASS_WEIGHTS = torch.tensor(
      sklearn.utils.class_weight.compute_class_weight(
        class_weight = 'balanced', 
        classes = np.unique(document_level_training_labels), 
        y = document_level_training_labels,
      ), dtype = torch.float).to(DEVICE)
  else:
    CLASS_WEIGHTS = None
  criterion = torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS, label_smoothing = label_smoothing)

  best_validation_performance = 0.0  
  best_validation_performance_loss = float('inf')
  best_training_performance_loss = float('inf')
  best_validation_performance_epoch = 0
  best_validation_loss = float('inf')
  best_validation_labels = list()
  best_validation_predictions = list()
  best_validation_probabilities = list()
  best_validation_identifiers = list()
  early_stopping_counter = 0

  epoch_runtimes = list()

  for epoch in range(epochs):

    model.train()
    total_loss = 0

    epoch_start_time = time.time()
    optimizer.zero_grad()

    for batch_i, batch in enumerate(training_batches):

      x = batch.x.to(DEVICE)
      edge_index = batch.edge_index.to(DEVICE)
      edge_attr = batch.edge_attr.to(DEVICE)
      _batch = batch.batch.to(DEVICE)
      y = batch.y.to(DEVICE)

      outputs = model(x, edge_index, edge_attr, _batch)
      
      loss = criterion(outputs, y)        
      total_loss += loss.item()
      if (ABLATION in [6, 7]) and (DATASET in ['R8', 'Ohsumed']):
        loss = loss / GRADIENT_ACCUMULATION_STEPS
      loss.backward()

      if (ABLATION in [6, 7]) and (DATASET in ['R8', 'Ohsumed']):
        if ((batch_i + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or (batch_i == len(training_batches) - 1):
          # Gradient clipping
          if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)        
          
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
      else:
        # Gradient clipping
        if gradient_clipping:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)        
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

      # Clear memory after each batch
      del x, edge_index, edge_attr, _batch, y, outputs
      gc.collect()
      torch.cuda.empty_cache()

    epoch_end_time = time.time()
    epoch_runtimes.append(epoch_end_time - epoch_start_time)
    average_training_loss = total_loss / len(training_batches)
    
    # ===========================================================================
    # ================================ Validating ===============================
    # ===========================================================================

    model.eval()
    total_validation_loss = 0
    validation_predictions = list()
    validation_labels = list()
    validation_identifiers = list()
    validation_probabilities = list()

    with torch.no_grad():
      for batch in validation_batches:
        
        x = batch.x.to(DEVICE)
        edge_index = batch.edge_index.to(DEVICE)
        edge_attr = batch.edge_attr.to(DEVICE)
        _batch = batch.batch.to(DEVICE)
        y = batch.y.to(DEVICE)
        identifier = batch.identifier.detach().cpu().numpy()
        
        outputs = model(x, edge_index, edge_attr, _batch)
        
        probabilities = torch.nn.functional.softmax(outputs, dim = 1)
        predictions = probabilities.argmax(dim = 1)

        loss = criterion(outputs, y)
        total_validation_loss += loss.item()

        validation_predictions.extend(predictions.detach().cpu().numpy())
        validation_labels.extend(y.detach().cpu().numpy())
        validation_identifiers.extend(identifier)
        validation_probabilities.extend([tuple(x) for x in probabilities.detach().cpu().numpy()])

        # Clear memory after each batch
        del x, edge_index, edge_attr, _batch, y, identifier, outputs, probabilities, predictions
        gc.collect()
        torch.cuda.empty_cache()
    
    average_validation_loss = total_validation_loss / len(validation_batches)
    
    validation_performance = sklearn.metrics.accuracy_score(validation_labels, validation_predictions) if ACCURACY else sklearn.metrics.f1_score(validation_labels, validation_predictions, average = 'macro')
    # print(f'[EPOCH {epoch}] Training loss: {average_training_loss} Validation loss: {average_validation_loss} Validation F1-score: {validation_performance}', flush = True)

    checkpointing_condition_met = (
        (average_validation_loss < best_validation_performance_loss)
        or ((average_validation_loss == best_validation_performance_loss) and (validation_performance > best_validation_performance))
      ) if CHECKPOINT_VALIDATION_LOSS else (
        (validation_performance > best_validation_performance)
        or ((validation_performance == best_validation_performance) and (average_validation_loss < best_validation_performance_loss))
      )
    if checkpointing_condition_met:
      best_validation_performance = validation_performance
      best_validation_performance_loss = average_validation_loss
      best_training_performance_loss = average_training_loss
      best_validation_performance_epoch = epoch + 1
      best_validation_labels = validation_labels.copy()
      best_validation_predictions = validation_predictions.copy()
      best_validation_probabilities = validation_probabilities.copy()
      best_validation_identifiers = validation_identifiers.copy()
      torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
      }, os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar'))

    # Early stopping
    if epoch >= early_stopping_start_epoch:
      if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        early_stopping_counter = 0
      else:
        early_stopping_counter += 1
        if early_stopping_counter > early_stopping_patience:
          break
    
    # Clear memory after each epoch
    gc.collect()
    torch.cuda.empty_cache()

  return model, best_validation_performance, \
    best_validation_performance_loss, \
    best_training_performance_loss, \
    best_validation_performance_epoch, \
    best_validation_labels, \
    best_validation_predictions, \
    best_validation_probabilities, \
    best_validation_identifiers, \
    epoch_runtimes


# ===========================================================================
# ============================== Model training =============================
# ===========================================================================

def train_and_predict(
    training_df,
    validation_df,
    testing_df,
    #
    random_state,
    #
    batch_size,
    #
    edge_threshold,
    node_threshold,
    #
    attention_heads,
    hidden_dimension,
    number_of_hidden_layers,
    dropout_rate,
    #
    learning_rate,
    weight_decay,
    #
    balanced_loss,
    epochs,
    #
    early_stopping_patience,
    early_stopping_start_epoch,
    #
    linear_warmup_step_ratio = None,
    linear_warmup_start_factor = None,
    linear_decay_end_factor = None,
    #
    label_smoothing = 0.0,
    gradient_clipping = None,
    #
    evaluate_test = False,
    trial_number = 0
  ):
      
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

  # ===========================================================================
  # =============================== Data loading ==============================
  # ===========================================================================

  try:

    os.makedirs(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'train'), exist_ok = True)
    data_sets.pre_construct_all_graphs_for_split(
      training_df,
      #
      data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'train'),
      #
      label_indices = np.unique(training_df['label'].values).tolist(),
      #
      chunk_size = 512,
      left_stride = 128,
      right_stride = 0,
      #
      token_threshold = 0.0 if ABLATION in [6, 7] else node_threshold,
      edge_threshold = 0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold),
      #
      plm = PLM,
      tokenizer = TOKENIZER,
      maximum_chunk_size = 512,
      #
      drop_first_level_edges_ablation = (ABLATION == 1),
      unit_weight_edges_ablation = (ABLATION == 2),
      drop_second_level_nodes_ablation = (ABLATION == 3),
      drop_second_and_third_level_nodes_ablation = (ABLATION == 4),
      bi_directional_edges_to_second_and_third_level_nodes_ablation = (ABLATION == 5),
      #
      device = DEVICE
    )

    training_dataset = data_sets.CachedGraphDataset(
      data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'train'),
      graph_count = training_df.shape[0],
    )

    os.makedirs(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'validation'), exist_ok = True)
    data_sets.pre_construct_all_graphs_for_split(
      validation_df,
      #
      data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'validation'),
      #
      label_indices = np.unique(training_df['label'].values).tolist(),
      #
      chunk_size = 512,
      left_stride = 128,
      right_stride = 0,
      #
      token_threshold = 0.0 if ABLATION in [6, 7] else node_threshold,
      edge_threshold = 0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold),
      #
      plm = PLM,
      tokenizer = TOKENIZER,
      maximum_chunk_size = 512,
      #
      drop_first_level_edges_ablation = (ABLATION == 1),
      unit_weight_edges_ablation = (ABLATION == 2),
      drop_second_level_nodes_ablation = (ABLATION == 3),
      drop_second_and_third_level_nodes_ablation = (ABLATION == 4),
      bi_directional_edges_to_second_and_third_level_nodes_ablation = (ABLATION == 5),
      #
      device = DEVICE
    )

    validation_dataset = data_sets.CachedGraphDataset(
      data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'validation'),
      graph_count = validation_df.shape[0],
    )
    
    # ===========================================================================
    # ============================== Batchification =============================
    # ===========================================================================

    training_batches = torch_geometric.loader.DataLoader(
      training_dataset,
      batch_size = int(batch_size / GRADIENT_ACCUMULATION_STEPS) if ((ABLATION in [6, 7]) and (DATASET in ['R8', 'Ohsumed'])) else batch_size,
      shuffle = True,
      generator = torch.Generator().manual_seed(random_state),
      num_workers = 4,
      persistent_workers = True,
      pin_memory = True,
      worker_init_fn = seed_worker
    )

    validation_batches = torch_geometric.loader.DataLoader(
      validation_dataset,
      batch_size = int(batch_size / GRADIENT_ACCUMULATION_STEPS) if ((ABLATION in [6, 7]) and (DATASET in ['R8', 'Ohsumed'])) else batch_size,
      shuffle = False,
      num_workers = 2,
      pin_memory = True,
      persistent_workers = True
    )
    
    # ===========================================================================
    # ================================= Training ================================
    # ===========================================================================

    model, best_validation_performance, best_validation_performance_loss, best_training_performance_loss, \
    best_validation_performance_epoch, best_validation_labels, best_validation_predictions, \
    best_validation_probabilities, best_validation_identifiers, epoch_runtimes = model_training(
      training_batches = training_batches,
      validation_batches = validation_batches,
      #
      attention_heads = attention_heads,
      hidden_dimension = hidden_dimension,
      number_of_hidden_layers = number_of_hidden_layers,
      dropout_rate = dropout_rate,
      #
      learning_rate = learning_rate,
      weight_decay = weight_decay,
      balanced_loss = balanced_loss,
      epochs = epochs,
      #
      early_stopping_patience = early_stopping_patience,
      early_stopping_start_epoch = early_stopping_start_epoch,
      #
      linear_warmup_step_ratio = linear_warmup_step_ratio,
      linear_warmup_start_factor = linear_warmup_start_factor,
      linear_decay_end_factor = linear_decay_end_factor,
      #
      label_smoothing = label_smoothing,
      gradient_clipping = gradient_clipping
    )

    # Clear memory
    shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'train'))
    shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'validation'))
    del training_batches
    del validation_batches
    gc.collect()
    torch.cuda.empty_cache()
    
    if not evaluate_test:
      # Remove model from GPU
      del model
      # Delete cache folder
      shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}'))
      
      return best_validation_performance, best_validation_performance_loss, best_training_performance_loss, best_validation_performance_epoch
    else:
      # ===========================================================================
      # ================================== Testing ================================
      # ===========================================================================

      os.makedirs(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'test'), exist_ok = True)
      data_sets.pre_construct_all_graphs_for_split(
        testing_df,
        #
        data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'test'),
        #
        label_indices = np.unique(training_df['label'].values).tolist(),
        #
        chunk_size = 512,
        left_stride = 128,
        right_stride = 0,
        #
        token_threshold = 0.0 if ABLATION in [6, 7] else node_threshold,
        edge_threshold = 0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold),
        #
        plm = PLM,
        tokenizer = TOKENIZER,
        maximum_chunk_size = 512,
        #
        drop_first_level_edges_ablation = (ABLATION == 1),
        unit_weight_edges_ablation = (ABLATION == 2),
        drop_second_level_nodes_ablation = (ABLATION == 3),
        drop_second_and_third_level_nodes_ablation = (ABLATION == 4),
        bi_directional_edges_to_second_and_third_level_nodes_ablation = (ABLATION == 5),
        #
        device = DEVICE
      )

      testing_dataset = data_sets.CachedGraphDataset(
        data_path = os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'test'),
        graph_count = testing_df.shape[0],
      )

      testing_batches = torch_geometric.loader.DataLoader(
        testing_dataset,
        batch_size = int(batch_size / GRADIENT_ACCUMULATION_STEPS) if ((ABLATION in [6, 7]) and (DATASET in ['R8', 'Ohsumed'])) else batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        persistent_workers = True
      )

      if os.path.exists(os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar')):
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar'), weights_only = False, map_location = 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
      
      # Store model locally
      os.makedirs(os.path.join(CACHE_PATH, DATASET, ABLATIONS[ABLATION], f'{trial_number}'), exist_ok = True)
      torch.save({
        'state_dict': model.state_dict()
      }, os.path.join(CACHE_PATH, DATASET, ABLATIONS[ABLATION], f'{trial_number}', f'{random_state}.pth.tar'))
      
      model.eval()
      test_predictions = list()
      test_labels = list()
      test_identifiers = list()
      test_probabilities = list()
      evaluation_runtime = 0
      with torch.no_grad():
        for batch in testing_batches:
          
          x = batch.x.to(DEVICE)
          edge_index = batch.edge_index.to(DEVICE)
          edge_attr = batch.edge_attr.to(DEVICE)
          _batch = batch.batch.to(DEVICE)
          y = batch.y.detach().cpu().numpy()
          identifier = batch.identifier.detach().cpu().numpy()

          evaluation_start_time = time.time()
          outputs = model(x, edge_index, edge_attr, _batch)
          probabilities = torch.nn.functional.softmax(outputs, dim = 1)
          predictions = probabilities.argmax(dim = 1)
          evaluation_runtime = evaluation_runtime + (time.time() - evaluation_start_time)

          test_predictions.extend(predictions.detach().cpu().numpy())
          test_labels.extend(y)
          test_identifiers.extend(identifier)
          test_probabilities.extend([tuple(x) for x in probabilities.detach().cpu().numpy()])

          # Clear memory after each batch
          del x, edge_index, edge_attr, _batch, y, identifier, outputs, probabilities, predictions
          gc.collect()
          torch.cuda.empty_cache()
      
      # Average evaluation time per instance
      average_evaluation_runtime = evaluation_runtime / testing_dataset.len()
      # Remove model from GPU
      del model
      # Clear memory
      shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}', 'test'))
      # Delete cache folder
      shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}'))
      del testing_batches
      gc.collect()
      torch.cuda.empty_cache()

      pd.DataFrame({
        'test_performance' : [sklearn.metrics.accuracy_score(test_labels, test_predictions) if ACCURACY else sklearn.metrics.f1_score(test_labels, test_predictions, average = 'macro')],
        'validation_performance' : [best_validation_performance],
        'test_average_evaluation_runtime' : [average_evaluation_runtime],
        'checkpoint_epoch' : [best_validation_performance_epoch],
        'random_state' : [random_state]
      }).to_csv(
        os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', 'metrics.csv'),
        mode = 'a',
        header = not os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', 'metrics.csv')),
        index = False
      )

      os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', f'{random_state}'), exist_ok = True)

      pd.DataFrame({
        'real' : test_labels + best_validation_labels,
        'prediction' : test_predictions + best_validation_predictions,
        'identifier' : test_identifiers + best_validation_identifiers,
        'split' : ['test'] * len(test_labels) + ['validation'] * len(best_validation_labels)
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', f'{random_state}', 'predictions.csv'), index = False)

      pd.DataFrame(
        test_probabilities + best_validation_probabilities
      ).assign(
        identifier = test_identifiers + best_validation_identifiers,
        split = ['test'] * len(test_labels) + ['validation'] * len(best_validation_labels)
      ).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', f'{random_state}', 'probabilities.csv'), index = False)

      pd.DataFrame({
        'epoch_runtime' : epoch_runtimes,
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial_number}', f'{random_state}', 'runtimes.csv'), index = False)

      return sklearn.metrics.accuracy_score(test_labels, test_predictions) if ACCURACY else sklearn.metrics.f1_score(test_labels, test_predictions, average = 'macro'), best_validation_performance, average_evaluation_runtime, best_validation_performance_epoch
  except Exception as e:
    print(e, flush = True)
    # Delete cache folder
    if os.path.isdir(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}')):
      shutil.rmtree(os.path.join(SCRATCH_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{0.0 if ABLATION in [0, 6] else (1.0 if ABLATION == 1 else edge_threshold)}-{0.0 if ABLATION in [6, 7] else node_threshold}', f'{RANDOM_STATE}'))
    return -1.0, float('inf'), float('inf'), 0

# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--use_label_smoothing', required = True, type = int, help = 'Whether or not to use label smoothing during training.')
  parser.add_argument('--use_gradient_clipping', required = True, type = int, help = 'Whether or not to use gradient clipping during training.')
  parser.add_argument('--checkpoint_validation_loss', required = True, type = int, help = 'Whether or not to use validation loss for checkpoints and early stopping. Uses the chosen performance metric if False.')
  parser.add_argument('--use_accuracy', required = True, type = int, help = 'Whether or not to use accuracy for model evaluation. Uses macro F1-score otherwise.')
  parser.add_argument('--use_balanced_loss', required = True, type = int, help = 'Whether or not to use balanced loss weights during model training.')
  parser.add_argument('--fine_tuning_trial_number', required = True, type = int, help = 'The number of the fine-tuned model to be used as basis for graph construction.')
  parser.add_argument('--ablation', required = True, type = int, help = 'The ablation to apply during graph construction. Choose an index from the following possibilities: [fully connected first-level, no first-level edges, unit weight edges, no second-level edges, no second- and third-level edges, bi-directional edges to second- and third-level nodes, no node and edge pruning].')
  parser.add_argument('--seed', required = True, type = int, help = 'The seed to use for all operations with random states.')

  args = parser.parse_args()
  DATASET = args.data_set
  LABEL_SMOOTHING = args.use_label_smoothing
  GRADIENT_CLIPPING = args.use_gradient_clipping
  CHECKPOINT_VALIDATION_LOSS = args.checkpoint_validation_loss
  ACCURACY = args.use_accuracy
  BALANCED_LOSS = args.use_balanced_loss
  FINE_TUNING_TRIAL_NUMBER = args.fine_tuning_trial_number
  ABLATION = args.ablation
  RANDOM_STATE = args.seed

  if LABEL_SMOOTHING not in [0, 1]:
    raise ValueError('The label smoothing parameter must be either 0 (False) or 1 (True).')
  LABEL_SMOOTHING = bool(LABEL_SMOOTHING)
  
  if GRADIENT_CLIPPING not in [0, 1]:
    raise ValueError('The gradient clipping parameter must be either 0 (False) or 1 (True).')

  if CHECKPOINT_VALIDATION_LOSS not in [0, 1]:
    raise ValueError('The checkpoint validation loss parameter must be either 0 (False) or 1 (True).')

  if ACCURACY not in [0, 1]:
    raise ValueError('The accuracy parameter must be either 0 (False) or 1 (True).')

  if BALANCED_LOSS not in [0, 1]:
    raise ValueError('The balanced loss parameter must be either 0 (False) or 1 (True).')

  if not os.path.exists(os.path.join('..', 'fine_tuning', 'models', DATASET, f'{FINE_TUNING_TRIAL_NUMBER}', f'{SEED}')):
    raise ValueError('The selected fine-tuning trial number does not contain a corresponding model stored in disk.')
  
  if (ABLATION < 0) or (ABLATION > 7):
    raise ValueError('The ablation parameter must be between 0 and 7.')

  GRADIENT_CLIPPING = bool(GRADIENT_CLIPPING)
  CHECKPOINT_VALIDATION_LOSS = bool(CHECKPOINT_VALIDATION_LOSS)
  ACCURACY = bool(ACCURACY)
  BALANCED_LOSS = bool(BALANCED_LOSS)

  HYPER_PARAMETERS = hyper_parameters.HYPER_PARAMETERS

  ABLATIONS = ['fully_connected_first_level_edges', 'drop_first_level_edges', 'unit_weight_edges', 'drop_second_level_nodes', 'drop_second_and_third_level_nodes', 'bi_directional_edges_to_second_and_third_level_nodes', 'no_node_and_edge_pruning', 'no_node_pruning']
  
  # GRADIENT_ACCUMULATION_STEPS = 28 if DATASET == 'Ohsumed' else 4 # Only used for no_node_and_edge_pruning ablation with R8 and Ohsumed

  TOKENIZER = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
  PLM = transformers.AutoModelForSequenceClassification.from_pretrained(
    os.path.join('..', 'fine_tuning', 'models', DATASET, f'{FINE_TUNING_TRIAL_NUMBER}', f'{SEED}'),
    output_attentions = True,
    attn_implementation = 'eager',
    output_hidden_states = True,
  )
  PLM.to(DEVICE)
  
  EMBEDDING_DIMENSION = PLM.config.hidden_size

  DATASET_CACHE_PATH = f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}'

  TRAINING_DF = pd.read_csv(os.path.join('..', 'data', 'with_validation_splits', DATASET, 'train.csv'))
  TRAINING_DF['label'] = TRAINING_DF['label'].astype(int)
  VALIDATION_DF = pd.read_csv(os.path.join('..', 'data', 'with_validation_splits', DATASET, 'validation.csv'))
  VALIDATION_DF['label'] = VALIDATION_DF['label'].astype(int)
  TESTING_DF = pd.read_csv(os.path.join('..', 'data', 'with_validation_splits', DATASET, 'test.csv'))
  TESTING_DF['label'] = TESTING_DF['label'].astype(int)
  
  UNBOUND_STUDY_NAME = f'{DATASET}-Chefer_importance-unbound'
  unbound_storage = f'sqlite:///../optuna_studies/{UNBOUND_STUDY_NAME}.db'

  unbound_study = optuna.load_study(
    study_name = UNBOUND_STUDY_NAME,
    storage = unbound_storage,
  )
  
  top_N_trials = unbound_study.trials_dataframe()
  top_N_trials = top_N_trials[
    ['number', 'value'] + 
    [col for col in top_N_trials if col.startswith('params_')] + 
    [col for col in top_N_trials if col.startswith('user_attrs_')]
  ]
  
  top_N_trials = top_N_trials.sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  top_N_max_key_length = max(len(column) for column in top_N_trials.columns)
  
  for trial in top_N_trials.itertuples(index = False):
    
    print('\n[TRIAL]', trial.number, '[VALIDATION PERFORMANCE]', trial.value, '[TRAINING LOSS]', trial.user_attrs_training_loss, '[VALIDATION LOSS]', trial.user_attrs_validation_loss, '\n', flush = True)
    
    for field in trial._fields:
      print(f'{field:<{top_N_max_key_length}}', '\t\t\t', f'{getattr(trial, field)}', flush = True)
    print('', flush = True)
  
    os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial.number}'), exist_ok = True)

    GRADIENT_ACCUMULATION_STEPS = trial.params_batch_size # Effective batch size = 1 for gradient accumulation 
  
    test_performances = list()
    validation_performances = list()

    for random_state in range(RANDOM_STATE - TEST_RUNS_AROUND_BASE_SEED[0], RANDOM_STATE + TEST_RUNS_AROUND_BASE_SEED[1] + 1):

      if os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial.number}', f'{random_state}', 'predictions.csv')) \
        and os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial.number}', f'{random_state}', 'probabilities.csv')):

        predictions = pd.read_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Chefer_importance-{ABLATIONS[ABLATION]}', f'{trial.number}', f'{random_state}', 'predictions.csv'))
        
        validation_predictions = predictions[predictions['split'] == 'validation'][['real', 'prediction']].rename(columns = {'real' : 'label'})
        validation_performance = sklearn.metrics.accuracy_score(validation_predictions['label'], validation_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(validation_predictions['label'], validation_predictions['prediction'], average = 'macro')
        
        test_predictions = predictions[predictions['split'] == 'test'][['real', 'prediction']].rename(columns = {'real' : 'label'})
        test_performance = sklearn.metrics.accuracy_score(test_predictions['label'], test_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(test_predictions['label'], test_predictions['prediction'], average = 'macro')
      else:

        test_performance, validation_performance, _, _ = train_and_predict(
          training_df = TRAINING_DF,
          validation_df = VALIDATION_DF,
          testing_df = TESTING_DF,
          #
          random_state = random_state,
          #
          batch_size = HYPER_PARAMETERS['batch_size']['value'] if HYPER_PARAMETERS['batch_size']['fixed'] else (trial.params_batch_size),
          #
          edge_threshold = HYPER_PARAMETERS['edge_threshold']['value'] if HYPER_PARAMETERS['edge_threshold']['fixed'] else (trial.params_edge_threshold),
          node_threshold = HYPER_PARAMETERS['node_threshold']['value'] if HYPER_PARAMETERS['node_threshold']['fixed'] else (trial.params_node_threshold),
          #
          attention_heads = HYPER_PARAMETERS['attention_heads']['value'] if HYPER_PARAMETERS['attention_heads']['fixed'] else (trial.params_attention_heads),
          hidden_dimension = HYPER_PARAMETERS['hidden_dimension']['value'] if HYPER_PARAMETERS['hidden_dimension']['fixed'] else (trial.params_hidden_dimension),
          number_of_hidden_layers = HYPER_PARAMETERS['number_of_hidden_layers']['value'] if HYPER_PARAMETERS['number_of_hidden_layers']['fixed'] else (trial.params_number_of_hidden_layers),
          dropout_rate = HYPER_PARAMETERS['dropout_rate']['value'] if HYPER_PARAMETERS['dropout_rate']['fixed'] else (trial.params_dropout_rate),
          #
          learning_rate = HYPER_PARAMETERS['learning_rate']['value'] if HYPER_PARAMETERS['learning_rate']['fixed'] else (trial.params_learning_rate),
          weight_decay = HYPER_PARAMETERS['weight_decay']['value'] if HYPER_PARAMETERS['weight_decay']['fixed'] else (trial.params_weight_decay),
          #
          balanced_loss = BALANCED_LOSS, # HYPER_PARAMETERS['balanced_loss']['value'] if HYPER_PARAMETERS['balanced_loss']['fixed'] else (trial.params_balanced_loss),
          epochs = HYPER_PARAMETERS['epochs']['value'] if HYPER_PARAMETERS['epochs']['fixed'] else (trial.params_epochs),
          #
          early_stopping_patience = HYPER_PARAMETERS['early_stopping_patience']['value'] if HYPER_PARAMETERS['early_stopping_patience']['fixed'] else (trial.params_early_stopping_patience),
          early_stopping_start_epoch = HYPER_PARAMETERS['early_stopping_start_epoch']['value'] if HYPER_PARAMETERS['early_stopping_start_epoch']['fixed'] else (trial.params_early_stopping_start_epoch),
          #
          linear_warmup_step_ratio = HYPER_PARAMETERS['linear_warmup_step_ratio']['value'] if HYPER_PARAMETERS['linear_warmup_step_ratio']['fixed'] else (trial.params_linear_warmup_step_ratio),
          linear_warmup_start_factor = HYPER_PARAMETERS['linear_warmup_start_factor']['value'] if HYPER_PARAMETERS['linear_warmup_start_factor']['fixed'] else (trial.params_linear_warmup_start_factor),
          linear_decay_end_factor = HYPER_PARAMETERS['linear_decay_end_factor']['value'] if HYPER_PARAMETERS['linear_decay_end_factor']['fixed'] else (trial.params_linear_decay_end_factor),
          #
          label_smoothing = HYPER_PARAMETERS['label_smoothing']['value'] if LABEL_SMOOTHING and HYPER_PARAMETERS['label_smoothing']['fixed'] else (trial.params_label_smoothing if LABEL_SMOOTHING else 0.0),
          gradient_clipping = HYPER_PARAMETERS['gradient_clipping']['value'] if GRADIENT_CLIPPING and HYPER_PARAMETERS['gradient_clipping']['fixed'] else (trial.params_gradient_clipping if GRADIENT_CLIPPING else None),
          #
          evaluate_test = True,
          trial_number = trial.number
        )
      
      if test_performance == -1.0:
        print(random_state, 'Exception...', flush = True)
        continue

      print(f'[{random_state}]', 'VALIDATION:', validation_performance, 'TESTING:', test_performance, flush = True)

      test_performances.append(test_performance)
      validation_performances.append(validation_performance)

    try:
      print('', flush = True)
      print(
        'Validation performance:',
        round(min(validation_performances) * 100, 2), '&',
        round(statistics.mean(validation_performances) * 100, 2), '±', round(statistics.stdev(validation_performances) * 100, 2), '&',
        round(max(validation_performances) * 100, 2),
        flush = True
      )
      print(
        'Testing performance:',
        round(min(test_performances) * 100, 2), '&',
        round(statistics.mean(test_performances) * 100, 2), '±', round(statistics.stdev(test_performances) * 100, 2), '&',
        round(max(test_performances) * 100, 2),
        flush = True
      )
    except:
      print('Not enough values to compute average and standard deviation performance.', flush = True)

  shutil.rmtree(CHECKPOINT_PATH)
  print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)