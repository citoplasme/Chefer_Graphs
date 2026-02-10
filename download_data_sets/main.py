import os
import requests
import torchtext.datasets

CACHE_PATH = '../data/as_is/'
os.makedirs(CACHE_PATH, exist_ok = True)

# IMDb
train_itererator, test_iterator = torchtext.datasets.IMDB(root = CACHE_PATH)
list(train_itererator) # forces the download
list(test_iterator)
print('[UPDATE] IMDB downloaded.', flush = True)

# SST-2
train_itererator, validation_iterator, test_iterator = torchtext.datasets.SST2(root = CACHE_PATH)
list(train_itererator)
list(validation_iterator)
list(test_iterator)
print('[UPDATE] SST2 downloaded.', flush = True)

# Reuters-21578
# Download zip from https://www.kaggle.com/datasets/thedevastator/uncovering-financial-insights-with-the-reuters-2

# Ohsumed
os.makedirs(os.path.join(CACHE_PATH, 'Ohsumed'), exist_ok = True)
urls = {
  'train.csv' : 'https://raw.githubusercontent.com/citoplasme/PLM_token_graphs/refs/heads/main/data/as_is/Ohsumed/train.csv',
  'test.csv' :  'https://raw.githubusercontent.com/citoplasme/PLM_token_graphs/refs/heads/main/data/as_is/Ohsumed/test.csv'
}
for filename, url in urls.items():
  response = requests.get(url)
  response.raise_for_status()
  with open(os.path.join(CACHE_PATH, 'Ohsumed', filename), 'wb') as file:
    file.write(response.content)
print('[UPDATE] Ohsumed downloaded.', flush = True)