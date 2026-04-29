import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import pandas as pd
import optuna
import random
import gc
import time
import sklearn.metrics
import sklearn.utils
import sklearn.model_selection
import argparse
import transformers
import statistics
import shutil

import hyper_parameters

import warnings
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()

st = time.time()

SEED = 42

TEST_RUNS_AROUND_BASE_SEED = (0, 0) #(5, 4)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STORAGE_PATH = '../../outputs/fine_tuning_downsampled/'
CACHE_PATH = os.path.join('.', 'models')
os.makedirs(CACHE_PATH, exist_ok = True)

CHECKPOINT_PATH = f'/scratch/jpimen/checkpoints/{os.environ["SLURM_JOB_ID"]}/'
os.makedirs(CHECKPOINT_PATH, exist_ok = True)

# ===========================================================================
# ================================ Data Set =================================
# ===========================================================================

# Based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_tokens, split):
    self.df = df
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.split = split
        
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    
    encoding = self.tokenizer.encode_plus(
      row['text'],
      add_special_tokens = True,
      max_length = self.max_tokens, # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification
      padding = 'max_length',
      truncation = True,
      return_attention_mask = True,
      return_tensors = 'pt',
    )
    return {
      'input_identifiers': encoding['input_ids'].flatten(),
      'attention_mask' : encoding['attention_mask'].flatten(),
      'y' : torch.tensor(row['label'], dtype = torch.long),
      'split' : self.split,
      'row' : torch.tensor(idx, dtype = torch.long),
      'identifier' : torch.tensor(row['identifier'], dtype = torch.long),
      'chunk' : torch.tensor(row['chunk'], dtype = torch.long)
    }

def subsample_training_split(df, percentage):
  identifier_label_df = df[['identifier', 'label']] \
    .drop_duplicates(subset = 'identifier') \
    .reset_index(drop = True)
  subsampled_df, _ = sklearn.model_selection.train_test_split(
    identifier_label_df,
    train_size = percentage,
    stratify = identifier_label_df['label'],
    random_state = SEED
  )
  return df[df['identifier'].isin(subsampled_df['identifier'])]

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
    tokenizer_length,
    #
    frozen_layers,
    classifier_dropout,
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
    
  model = transformers.AutoModelForSequenceClassification.from_pretrained(
    'google-bert/bert-base-uncased',
    output_attentions = True,
    attn_implementation = 'eager',
    output_hidden_states = True,
    num_labels = np.unique(pd.read_csv(os.path.join('..', '..', 'data', 'with_validation_splits', DATASET, 'train.csv'))['label'].values).size
  )
  model.classifier.dropout = torch.nn.Dropout(classifier_dropout)
  model.to(DEVICE)

  layer_count = model.config.num_hidden_layers
  if (frozen_layers < 0) or (frozen_layers > layer_count):
    raise ValueError(f'The number of frozen layers must be a value between 0 and {layer_count} for model "google-bert/bert-base-uncased". Received {frozen_layers}.')
  # Freeze model layers
  for layer in model.bert.encoder.layer[:frozen_layers]:
    for param in layer.parameters():
      param.requires_grad = False
  # Freeze embeddings
  # for param in model.bert.embeddings.parameters():
  #   param.requires_grad = False
  
  decay_params = list()
  no_decay_params = list()
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if any(nd in name for nd in ['bias', 'LayerNorm']):
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
    document_level_training_labels = pd.read_csv(os.path.join('..', '..', 'data', 'with_validation_splits', DATASET, 'train.csv'))['label'].values
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
  best_validation_rows = list()
  best_validation_identifiers = list()
  best_validation_chunks = list()
  early_stopping_counter = 0

  epoch_runtimes = list()

  for epoch in range(epochs):

    model.train()
    total_loss = 0

    epoch_start_time = time.time()
    optimizer.zero_grad()

    for batch_i, batch in enumerate(training_batches):

      input_identifiers = batch['input_identifiers'].to(DEVICE)
      attention_mask = batch['attention_mask'].to(DEVICE)
      y = batch['y'].to(DEVICE)
      
      outputs = model(input_identifiers, attention_mask = attention_mask)
        
      loss = criterion(outputs.logits, y)
      total_loss += loss.item()
      loss.backward()
      
      # Gradient clipping
      if gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradient_clipping)        

      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

      # Clear memory after each batch
      del input_identifiers, attention_mask, y, outputs
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
    validation_rows = list()
    validation_identifiers = list()
    validation_chunks = list()
    validation_probabilities = list()

    with torch.no_grad():
      for batch in validation_batches:

        input_identifiers = batch['input_identifiers'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        y = batch['y'].to(DEVICE)
        rows = batch['row']
        identifiers = batch['identifier']
        chunks = batch['chunk']
        
        outputs = model(input_identifiers, attention_mask = attention_mask)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim = 1)
        predictions = probabilities.argmax(dim = 1)

        loss = criterion(outputs.logits, y)
        total_validation_loss += loss.item()

        validation_predictions.extend(predictions.detach().cpu().numpy())
        validation_labels.extend(y.detach().cpu().numpy())
        validation_identifiers.extend(identifiers.detach().cpu().numpy())
        validation_rows.extend(rows.detach().cpu().numpy())
        validation_chunks.extend(chunks.detach().cpu().numpy())
        validation_probabilities.extend([tuple(x) for x in probabilities.detach().cpu().numpy()])

        # Clear memory after each batch
        del input_identifiers, attention_mask, y, outputs, probabilities, predictions
        gc.collect()
        torch.cuda.empty_cache()
    
    class_probability_columns = [str(x) for x in list(range(len(validation_probabilities[0])))]
    validation_document_level_predictions = pd.DataFrame(
      validation_probabilities,
      columns = class_probability_columns
    ).assign(
      identifier = validation_identifiers,
      label = validation_labels
    ).groupby('identifier', as_index = False) \
      .agg(
        **{c : (c, 'mean') for c in class_probability_columns},
        label = ('label', 'first')
      )
    validation_document_level_predictions['prediction'] = (
      validation_document_level_predictions[class_probability_columns]
        .values
        .argmax(axis = 1)
    )

    average_validation_loss = total_validation_loss / len(validation_batches)
    validation_performance = sklearn.metrics.accuracy_score(validation_document_level_predictions['label'], validation_document_level_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(validation_document_level_predictions['label'], validation_document_level_predictions['prediction'], average = 'macro')
    # print(f'[{time.time()}] [EPOCH {epoch}] Training loss: {average_training_loss} Validation loss: {average_validation_loss} Validation {"F1-score" if not ACCURACY else "accuracy"}: {validation_performance}', flush = True)

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
      best_validation_rows = validation_rows.copy()
      best_validation_identifiers = validation_identifiers.copy()
      best_validation_chunks = validation_chunks.copy()
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
    best_validation_rows, \
    best_validation_identifiers, \
    best_validation_chunks, \
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
    frozen_layers,
    classifier_dropout,
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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

    training_dataset = CustomDataset(
      df = training_df,
      tokenizer = tokenizer,
      max_tokens = 512,
      split = 'training'
    )

    validation_dataset = CustomDataset(
      df = validation_df,
      tokenizer = tokenizer,
      max_tokens = 512,
      split = 'validation'
    )
    
    # ===========================================================================
    # ============================== Batchification =============================
    # ===========================================================================

    training_batches = torch.utils.data.DataLoader(
      training_dataset,
      batch_size = batch_size,
      shuffle = True,
      generator = torch.Generator().manual_seed(random_state),
      num_workers = 4,
      persistent_workers = True,
      pin_memory = True,
      worker_init_fn = seed_worker
    )

    validation_batches = torch.utils.data.DataLoader(
      validation_dataset,
      batch_size = batch_size,
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
    best_validation_probabilities, best_validation_rows, best_validation_identifiers, \
    best_validation_chunks, epoch_runtimes = model_training(
      training_batches = training_batches,
      validation_batches = validation_batches,
      #
      tokenizer_length = len(tokenizer),
      #
      frozen_layers = frozen_layers,
      classifier_dropout = classifier_dropout,
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
    del training_batches
    del validation_batches
    gc.collect()
    torch.cuda.empty_cache()
    
    if not evaluate_test:
      # Remove model from GPU
      del model      
      return best_validation_performance, best_validation_performance_loss, best_training_performance_loss, best_validation_performance_epoch
    else:
      # ===========================================================================
      # ================================== Testing ================================
      # ===========================================================================

      testing_dataset = CustomDataset(
        df = testing_df,
        tokenizer = tokenizer,
        max_tokens = 512,
        split = 'testing'
      )
      
      testing_batches = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size = batch_size,
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
      
      # Store model locally for Chefer-based graph construction
      os.makedirs(os.path.join(CACHE_PATH, DATASET, f'{trial_number}', f'{SUBSAMPLE_PERCENTAGE}'), exist_ok = True)
      model.save_pretrained(os.path.join(CACHE_PATH, DATASET, f'{trial_number}', f'{SUBSAMPLE_PERCENTAGE}', f'{random_state}'))

      model.eval()
      test_predictions = list()
      test_labels = list()
      test_rows = list()
      test_identifiers = list()
      test_chunks = list()
      test_probabilities = list()
      evaluation_runtime = 0
      with torch.no_grad():
        for batch in testing_batches:
          
          input_identifiers = batch['input_identifiers'].to(DEVICE)
          attention_mask = batch['attention_mask'].to(DEVICE)
          y = batch['y'].to(DEVICE)
          rows = batch['row']
          identifiers = batch['identifier']
          chunks = batch['chunk']
          
          evaluation_start_time = time.time()
          outputs = model(input_identifiers, attention_mask = attention_mask)
          probabilities = torch.nn.functional.softmax(outputs.logits, dim = 1)
          predictions = probabilities.argmax(dim = 1)
          evaluation_runtime = evaluation_runtime + (time.time() - evaluation_start_time)

          test_predictions.extend(predictions.detach().cpu().numpy())
          test_labels.extend(y.detach().cpu().numpy())
          test_rows.extend(rows.detach().cpu().numpy())
          test_identifiers.extend(identifiers.detach().cpu().numpy())
          test_chunks.extend(chunks.detach().cpu().numpy())
          test_probabilities.extend([tuple(x) for x in probabilities.detach().cpu().numpy()])

          # Clear memory after each batch
          del input_identifiers, attention_mask, y, outputs, probabilities, predictions
          gc.collect()
          torch.cuda.empty_cache()
      
      # Average evaluation time per instance
      average_evaluation_runtime = evaluation_runtime / testing_dataset.__len__()
      # Remove model from GPU
      del model
      # Clear memory
      del testing_batches
      gc.collect()
      torch.cuda.empty_cache()

      class_probability_columns = [str(x) for x in list(range(len(test_probabilities[0])))]
      test_document_level_predictions = pd.DataFrame(
        test_probabilities,
        columns = class_probability_columns
      ).assign(
        identifier = test_identifiers,
        label = test_labels
      ).groupby('identifier', as_index = False) \
        .agg(
          **{c : (c, 'mean') for c in class_probability_columns},
          label = ('label', 'first')
        )
      test_document_level_predictions['prediction'] = (
        test_document_level_predictions[class_probability_columns]
          .values
          .argmax(axis = 1)
      )

      pd.DataFrame({
        'test_performance' : [sklearn.metrics.accuracy_score(test_document_level_predictions['label'], test_document_level_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(test_document_level_predictions['label'], test_document_level_predictions['prediction'], average = 'macro')],
        'validation_performance' : [best_validation_performance],
        'test_average_evaluation_runtime' : [average_evaluation_runtime],
        'checkpoint_epoch' : [best_validation_performance_epoch],
        'random_state' : [random_state]
      }).to_csv(
        os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', 'metrics.csv'),
        mode = 'a',
        header = not os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', 'metrics.csv')),
        index = False
      )
      
      os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', f'{random_state}'), exist_ok = True)

      pd.DataFrame({
        'real' : test_labels + best_validation_labels,
        'prediction' : test_predictions + best_validation_predictions,
        'index' : test_rows + best_validation_rows,
        'identifier' : test_identifiers + best_validation_identifiers,
        'chunk' : test_chunks + best_validation_chunks,
        'split' : ['test'] * len(test_labels) + ['validation'] * len(best_validation_labels)
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', f'{random_state}', 'predictions.csv'), index = False)

      pd.DataFrame(
        test_probabilities + best_validation_probabilities
      ).assign(
        index = test_rows + best_validation_rows,
        identifier = test_identifiers + best_validation_identifiers,
        chunk = test_chunks + best_validation_chunks,
        split = ['test'] * len(test_labels) + ['validation'] * len(best_validation_labels)
      ).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', f'{random_state}', 'probabilities.csv'), index = False)

      pd.DataFrame({
        'epoch_runtime' : epoch_runtimes,
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial_number}', f'{random_state}', 'runtimes.csv'), index = False)

      return sklearn.metrics.accuracy_score(test_document_level_predictions['label'], test_document_level_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(test_document_level_predictions['label'], test_document_level_predictions['prediction'], average = 'macro'), best_validation_performance, average_evaluation_runtime, best_validation_performance_epoch
  except Exception as e:
    print(e, flush = True)
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

  args = parser.parse_args()
  DATASET = args.data_set
  LABEL_SMOOTHING = args.use_label_smoothing
  GRADIENT_CLIPPING = args.use_gradient_clipping
  CHECKPOINT_VALIDATION_LOSS = args.checkpoint_validation_loss
  ACCURACY = args.use_accuracy
  BALANCED_LOSS = args.use_balanced_loss
  FINE_TUNING_TRIAL_NUMBER = args.fine_tuning_trial_number

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
  
  GRADIENT_CLIPPING = bool(GRADIENT_CLIPPING)
  CHECKPOINT_VALIDATION_LOSS = bool(CHECKPOINT_VALIDATION_LOSS)
  ACCURACY = bool(ACCURACY)
  BALANCED_LOSS = bool(BALANCED_LOSS)

  HYPER_PARAMETERS = hyper_parameters.HYPER_PARAMETERS

  TRAINING_DF = pd.read_csv(os.path.join('..', '..', 'data', 'chunked_documents', DATASET, 'train.csv'))
  VALIDATION_DF = pd.read_csv(os.path.join('..', '..', 'data', 'chunked_documents', DATASET, 'validation.csv'))
  TESTING_DF = pd.read_csv(os.path.join('..', '..', 'data', 'chunked_documents', DATASET, 'test.csv'))
    
  UNBOUND_STUDY_NAME = f'{DATASET}-Fine-tuning-unbound'
  unbound_storage = f'sqlite:///../../optuna_studies/{UNBOUND_STUDY_NAME}.db'
  unbound_study = optuna.load_study(
    study_name = UNBOUND_STUDY_NAME,
    storage = unbound_storage
  )
  
  top_N_trials = unbound_study.trials_dataframe()
  top_N_trials = top_N_trials[top_N_trials['number'] == FINE_TUNING_TRIAL_NUMBER]
  
  top_N_max_key_length = max(len(column) for column in top_N_trials.columns)
  
  for trial in top_N_trials.itertuples(index = False):
    
    print('\n[TRIAL]', trial.number, '[VALIDATION PERFORMANCE]', trial.value, '[TRAINING LOSS]', trial.user_attrs_training_loss, '[VALIDATION LOSS]', trial.user_attrs_validation_loss, '\n', flush = True)
    
    for field in trial._fields:
      print(f'{field:<{top_N_max_key_length}}', '\t\t\t', f'{getattr(trial, field)}', flush = True)
    print('', flush = True)

    for SUBSAMPLE_PERCENTAGE in [0.75, 0.5, 0.25]:
      print(f'{"subsample_percentage":<{top_N_max_key_length}}', '\t\t\t', f'{SUBSAMPLE_PERCENTAGE}', flush = True)
      print('', flush = True)
      
      os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial.number}'), exist_ok = True)
    
      test_performances = list()
      validation_performances = list()

      for random_state in range(SEED - TEST_RUNS_AROUND_BASE_SEED[0], SEED + TEST_RUNS_AROUND_BASE_SEED[1] + 1):
        if os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial.number}', f'{random_state}', 'predictions.csv')) \
          and os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial.number}', f'{random_state}', 'probabilities.csv')):

          predictions = pd.read_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial.number}', f'{random_state}', 'predictions.csv'))
          probabilities = pd.read_csv(os.path.join(STORAGE_PATH, f'{DATASET}-Fine-tuning-{SUBSAMPLE_PERCENTAGE}', f'{trial.number}', f'{random_state}', 'probabilities.csv'))
          
          class_probability_columns = [str(x) for x in range(probabilities.shape[1] - 4)] # index, split, identifier, chunk 
          
          validation_predictions = predictions[predictions['split'] == 'validation'][['identifier', 'chunk', 'real']].rename(columns = {'real' : 'label'})
          validation_probabilities = probabilities[probabilities['split'] == 'validation'][['identifier', 'chunk'] + class_probability_columns]
          
          validation_document_level_predictions = validation_predictions.merge(
            validation_probabilities,
            on = ['identifier', 'chunk']
          ).groupby('identifier', as_index = False) \
          .agg(
            **{c : (c, 'mean') for c in class_probability_columns},
            label = ('label', 'first')
          )
          validation_document_level_predictions['prediction'] = (
            validation_document_level_predictions[class_probability_columns]
              .values
              .argmax(axis = 1)
          )

          validation_performance = sklearn.metrics.accuracy_score(validation_document_level_predictions['label'], validation_document_level_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(validation_document_level_predictions['label'], validation_document_level_predictions['prediction'], average = 'macro')
          
          test_predictions = predictions[predictions['split'] == 'test'][['identifier', 'chunk', 'real']].rename(columns = {'real' : 'label'})
          test_probabilities = probabilities[probabilities['split'] == 'test'][['identifier', 'chunk'] + class_probability_columns]

          test_document_level_predictions = test_predictions.merge(
            test_probabilities,
            on = ['identifier', 'chunk']
          ).groupby('identifier', as_index = False) \
          .agg(
            **{c : (c, 'mean') for c in class_probability_columns},
            label = ('label', 'first')
          )
          test_document_level_predictions['prediction'] = (
            test_document_level_predictions[class_probability_columns]
              .values
              .argmax(axis = 1)
          )

          test_performance = sklearn.metrics.accuracy_score(test_document_level_predictions['label'], test_document_level_predictions['prediction']) if ACCURACY else sklearn.metrics.f1_score(test_document_level_predictions['label'], test_document_level_predictions['prediction'], average = 'macro')
        else:
          test_performance, validation_performance, _, _ = train_and_predict(
            training_df = subsample_training_split(df = TRAINING_DF, percentage = SUBSAMPLE_PERCENTAGE), # Sub-sample training dataframe
            validation_df = VALIDATION_DF,
            testing_df = TESTING_DF,
            #
            random_state = random_state,
            #
            batch_size = HYPER_PARAMETERS['batch_size']['value'] if HYPER_PARAMETERS['batch_size']['fixed'] else (trial.params_batch_size),
            #
            frozen_layers = HYPER_PARAMETERS['frozen_layers']['value'] if HYPER_PARAMETERS['frozen_layers']['fixed'] else (trial.params_frozen_layers),
            classifier_dropout = HYPER_PARAMETERS['classifier_dropout']['value'] if HYPER_PARAMETERS['classifier_dropout']['fixed'] else (trial.params_classifier_dropout),
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