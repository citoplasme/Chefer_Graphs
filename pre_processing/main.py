import os
import re
import pandas as pd
import time
import sklearn.model_selection

# ===========================================================================
# ============================ Global parameters ============================
# ===========================================================================

ORIGINAL_DATA_SET_PATH = '../data/as_is/'
PRE_PROCESSED_PATH = '../data/with_validation_splits/'

TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

SEED = 42

VALIDATION_SIZE = 0.2

# ===========================================================================
# ========================== Dataset standardization ========================
# ===========================================================================

def SST_2():

  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'SST-2'), exist_ok = True)

  # Training split
  pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'SST', 'train.csv'))[['sentence', 'binary_label']] \
    .dropna() \
    .rename(columns = {'binary_label': LABEL_COLUMN, 'sentence' : TEXT_COLUMN}) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'SST-2', 'train.csv'), index = False)
  
  # Validation split
  pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'SST', 'validation.csv'))[['sentence', 'binary_label']] \
    .dropna() \
    .rename(columns = {'binary_label': LABEL_COLUMN, 'sentence' : TEXT_COLUMN}) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'SST-2', 'validation.csv'), index = False)
  
  # Testing split
  pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'SST', 'test.csv'))[['sentence', 'binary_label']] \
    .dropna() \
    .rename(columns = {'binary_label': LABEL_COLUMN, 'sentence' : TEXT_COLUMN}) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'SST-2', 'test.csv'), index = False)

def IMDb():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'IMDb'), exist_ok = True)

  html_tags_regex = re.compile('<.*?>')

  # Training split
  documents = list()
  for filename, label in [('neg', 0), ('pos', 1)]:
    with open(os.path.join(ORIGINAL_DATA_SET_PATH, 'IMDb', 'train', filename), 'r', encoding = 'utf-8') as file:
      for line in file:
        line = re.sub(html_tags_regex, '', line.strip())
        if line:
          documents.append((line, label))
  training_df = pd.DataFrame(documents, columns = [TEXT_COLUMN, LABEL_COLUMN])
    
  # Validation split
  training_df, validation_df = sklearn.model_selection.train_test_split(
    training_df,
    test_size = VALIDATION_SIZE,
    random_state = SEED,
    shuffle = True,
    stratify = training_df[LABEL_COLUMN]
  )

  training_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'IMDb', 'train.csv'), index = False)
  
  validation_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'IMDb', 'validation.csv'), index = False)
  
  # Testing split
  documents = list()
  for filename, label in [('neg', 0), ('pos', 1)]:
    with open(os.path.join(ORIGINAL_DATA_SET_PATH, 'IMDb', 'test', filename), 'r', encoding = 'utf-8') as file:
      for line in file:
        line = re.sub(html_tags_regex, '', line.strip())
        if line:
          documents.append((line, label))
  pd.DataFrame(documents, columns = [TEXT_COLUMN, LABEL_COLUMN]) \
    .dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'IMDb', 'test.csv'), index = False)

def AGNews():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'AGNews'), exist_ok = True)

  # Training split
  training_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'AGNEWS', 'train.csv'), header = None, names = [LABEL_COLUMN, 'title', 'content'])
  training_df[LABEL_COLUMN] = training_df[LABEL_COLUMN].astype(int) - 1
  training_df[TEXT_COLUMN] = '[TITLE] ' + training_df['title'] + ' [CONTENT] ' + training_df['content']

  # Validation split
  training_df, validation_df = sklearn.model_selection.train_test_split(
    training_df,
    test_size = VALIDATION_SIZE,
    random_state = SEED,
    shuffle = True,
    stratify = training_df[LABEL_COLUMN]
  )

  training_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'AGNews', 'train.csv'), index = False)
  
  validation_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'AGNews', 'validation.csv'), index = False)
  
  # Testing split
  testing_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'AGNEWS', 'test.csv'), header = None, names = [LABEL_COLUMN, 'title', 'content'])
  testing_df[LABEL_COLUMN] = testing_df[LABEL_COLUMN].astype(int) - 1
  testing_df[TEXT_COLUMN] = '[TITLE] ' + testing_df['title'] + ' [CONTENT] ' + testing_df['content']
  testing_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'AGNews', 'test.csv'), index = False)

def Ohsumed():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed'), exist_ok = True)

  # Training split
  documents = list()
  for folder in os.listdir(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'training')):
    label = int(folder.replace('C', '')) - 1
    for filename in os.listdir(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'training', folder)):
      with open(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'training', folder, filename), 'r', encoding = 'utf-8') as file:
        text = file.read().strip()
        documents.append((text, label))
  training_df = pd.DataFrame(documents, columns = [TEXT_COLUMN, LABEL_COLUMN])
    
  # Validation split
  training_df, validation_df = sklearn.model_selection.train_test_split(
    training_df,
    test_size = VALIDATION_SIZE,
    random_state = SEED,
    shuffle = True,
    stratify = training_df[LABEL_COLUMN]
  )

  training_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed', 'train.csv'), index = False)
  
  validation_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed', 'validation.csv'), index = False)
  
  # Testing split
  documents = list()
  for folder in os.listdir(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'test')):
    label = int(folder.replace('C', '')) - 1
    for filename in os.listdir(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'test', folder)):
      with open(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'test', folder, filename), 'r', encoding = 'utf-8') as file:
        text = file.read().strip()
        documents.append((text, label))
  pd.DataFrame(documents, columns = [TEXT_COLUMN, LABEL_COLUMN]) \
    .dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed', 'test.csv'), index = False)

def DBPedia():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'DBPedia'), exist_ok = True)

  # Training split
  training_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'DBPedia', 'train.csv'), header = None, names = [LABEL_COLUMN, 'title', 'content'])
  training_df[LABEL_COLUMN] = training_df[LABEL_COLUMN].astype(int) - 1
  training_df[TEXT_COLUMN] = '[TITLE] ' + training_df['title'] + ' [CONTENT] ' + training_df['content']

  # Validation split
  training_df, validation_df = sklearn.model_selection.train_test_split(
    training_df,
    test_size = VALIDATION_SIZE,
    random_state = SEED,
    shuffle = True,
    stratify = training_df[LABEL_COLUMN]
  )

  training_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'DBPedia', 'train.csv'), index = False)
  
  validation_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'DBPedia', 'validation.csv'), index = False)
  
  # Testing split
  testing_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'DBPedia', 'test.csv'), header = None, names = [LABEL_COLUMN, 'title', 'content'])
  testing_df[LABEL_COLUMN] = testing_df[LABEL_COLUMN].astype(int) - 1
  testing_df[TEXT_COLUMN] = '[TITLE] ' + testing_df['title'] + ' [CONTENT] ' + testing_df['content']
  testing_df.dropna() \
    .drop(columns = ['title', 'content']) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'DBPedia', 'test.csv'), index = False)

if __name__ == '__main__':
  st = time.time()
  SST_2()
  print('[SST-2] Elapsed time:', time.time() - st, 'seconds.', flush = True)
  
  st = time.time()
  IMDb()
  print('[IMDb] Elapsed time:', time.time() - st, 'seconds.', flush = True)

  st = time.time()
  AGNews()
  print('[AGNews] Elapsed time:', time.time() - st, 'seconds.', flush = True)

  st = time.time()
  DBPedia()
  print('[DBPedia] Elapsed time:', time.time() - st, 'seconds.', flush = True)

  st = time.time()
  Ohsumed()
  print('[Ohsumed] Elapsed time:', time.time() - st, 'seconds.', flush = True)