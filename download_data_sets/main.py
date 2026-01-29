import os
import io
import tarfile
import urllib.request
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

# DBPedia
# train_itererator, test_iterator = torchtext.datasets.DBpedia(root = CACHE_PATH)
# list(train_itererator)
# list(test_iterator)
# Must be manually downloaded from 'https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&confirm=t'
# print('[UPDATE] DBpedia downloaded.', flush = True)

# AGNews
train_itererator, test_iterator = torchtext.datasets.AG_NEWS(root = CACHE_PATH)
list(train_itererator)
list(test_iterator)
print('[UPDATE] AGNews downloaded.', flush = True)

# Ohsumed
url = 'http://disi.unitn.it/moschitti/corpora/ohsumed-first-20000-docs.tar.gz'
with urllib.request.urlopen(url) as response:
  file_object = io.BytesIO(response.read())
  with tarfile.open(fileobj = file_object, mode = 'r:gz') as tar:
    tar.extractall(CACHE_PATH, filter = 'data')
os.rename(os.path.join(CACHE_PATH, 'ohsumed-first-20000-docs'), os.path.join(CACHE_PATH, 'Ohsumed'))
print('[UPDATE] Ohsumed downloaded.', flush = True)