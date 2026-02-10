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

SEED = 38

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

gc.collect()
torch.cuda.empty_cache()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(SEED)
torch.use_deterministic_algorithms(mode = True)

tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

testing_df = pd.read_csv(os.path.join('..', 'data', 'chunked_documents', 'SST-2', 'test.csv'))

testing_dataset = CustomDataset(
  df = testing_df,
  tokenizer = tokenizer,
  max_tokens = 512,
  split = 'testing'
)

testing_batches = torch.utils.data.DataLoader(
  testing_dataset,
  batch_size = 32,
  shuffle = False,
  num_workers = 2,
  pin_memory = True, 
  persistent_workers = True
)

CACHE_PATH = os.path.join('.', 'models')
model = transformers.AutoModelForSequenceClassification.from_pretrained(
  os.path.join(CACHE_PATH, 'SST-2', f'{0}', f'{SEED}'),
  output_attentions = True,
  attn_implementation = 'eager',
  output_hidden_states = True,
)
model.to(DEVICE)

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
    outputs = model(input_identifiers, attention_mask = attention_mask, labels = y)
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

print(sklearn.metrics.accuracy_score(test_document_level_predictions['label'], test_document_level_predictions['prediction']), flush = True)