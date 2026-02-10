import os
import re
import pandas as pd
import time
import sklearn.model_selection
import ast
import html

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

def R8():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'R8'), exist_ok = True)
  
  topics = [(0, 'acq'), (1, 'crude'), (2, 'earn'), (3, 'grain'), (4, 'interest'), (5, 'money-fx'), (6, 'ship'), (7, 'trade')]
  topic_to_integer = {name : idx for idx, name in topics}

  # Training split
  training_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'Reuters-21578', 'ModApte_train.csv'))[[TEXT_COLUMN, 'topics']]
  training_df['topics'] = training_df['topics'].apply(ast.literal_eval)
  training_df = training_df[training_df['topics'].apply(len) == 1]
  training_df = training_df[training_df['topics'].apply(lambda x : x[0] in [t[1] for t in topics])]
  training_df[LABEL_COLUMN] = training_df['topics'].apply(lambda x : topic_to_integer[x[0]])
  training_df = training_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
  training_df = training_df[training_df[TEXT_COLUMN].str.strip() != '']

  training_df[TEXT_COLUMN] = training_df[TEXT_COLUMN].str.replace(r'reuter\^?m?', '', regex = True, flags = re.IGNORECASE)
  training_df[TEXT_COLUMN] = training_df[TEXT_COLUMN].apply(html.unescape)
  training_df[TEXT_COLUMN] = training_df[TEXT_COLUMN].str.replace(r'\s+', ' ', regex = True).str.strip()

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
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'R8', 'train.csv'), index = False)
  
  validation_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'R8', 'validation.csv'), index = False)

  # Testing split
  testing_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'Reuters-21578', 'ModApte_test.csv'))[[TEXT_COLUMN, 'topics']]
  testing_df['topics'] = testing_df['topics'].apply(ast.literal_eval)
  testing_df = testing_df[testing_df['topics'].apply(len) == 1]
  testing_df = testing_df[testing_df['topics'].apply(lambda x : x[0] in [t[1] for t in topics])]
  testing_df[LABEL_COLUMN] = testing_df['topics'].apply(lambda x : topic_to_integer[x[0]])
  testing_df = testing_df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
  testing_df = testing_df[testing_df[TEXT_COLUMN].str.strip() != '']

  testing_df[TEXT_COLUMN] = testing_df[TEXT_COLUMN].str.replace(r'reuter\^?m?', '', regex = True, flags = re.IGNORECASE)
  testing_df[TEXT_COLUMN] = testing_df[TEXT_COLUMN].apply(html.unescape)
  testing_df[TEXT_COLUMN] = testing_df[TEXT_COLUMN].str.replace(r'\s+', ' ', regex = True).str.strip()

  testing_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'R8', 'test.csv'), index = False)

def Ohsumed():
  os.makedirs(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed'), exist_ok = True)

  # Training split
  training_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'train.csv'))[[TEXT_COLUMN, LABEL_COLUMN]]
  training_df[LABEL_COLUMN] = training_df[LABEL_COLUMN].map(lambda x : x.replace('C', '')).astype(int) - 1
    
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
  testing_df = pd.read_csv(os.path.join(ORIGINAL_DATA_SET_PATH, 'Ohsumed', 'test.csv'))[[TEXT_COLUMN, LABEL_COLUMN]]
  testing_df[LABEL_COLUMN] = testing_df[LABEL_COLUMN].map(lambda x : x.replace('C', '')).astype(int) - 1
  
  testing_df.dropna() \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(PRE_PROCESSED_PATH, 'Ohsumed', 'test.csv'), index = False)

if __name__ == '__main__':
  # st = time.time()
  # SST_2()
  # print('[SST-2] Elapsed time:', time.time() - st, 'seconds.', flush = True)
  
  # st = time.time()
  # IMDb()
  # print('[IMDb] Elapsed time:', time.time() - st, 'seconds.', flush = True)

  st = time.time()
  R8()
  print('[R8] Elapsed time:', time.time() - st, 'seconds.', flush = True)

  # st = time.time()
  # Ohsumed()
  # print('[Ohsumed] Elapsed time:', time.time() - st, 'seconds.', flush = True)