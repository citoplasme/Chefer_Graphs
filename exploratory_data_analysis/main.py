import os
import pandas as pd

CACHE_PATH = '../data/with_validation_splits/'

def get_data_set_properties(data_set):
  training_df = pd.read_csv(os.path.join(CACHE_PATH, data_set, 'train.csv'))
  validation_df = pd.read_csv(os.path.join(CACHE_PATH, data_set, 'validation.csv'))
  testing_df = pd.read_csv(os.path.join(CACHE_PATH, data_set, 'test.csv'))

  print(training_df.shape, validation_df.shape, testing_df.shape, flush = True)

if __name__ == '__main__':
  get_data_set_properties('SST-2')

  get_data_set_properties('IMDb')

  get_data_set_properties('AGNews')

  get_data_set_properties('DBPedia')

  get_data_set_properties('Ohsumed')



# import pandas as pd
# import plotnine as p9
# from sklearn.feature_extraction.text import CountVectorizer
# import textstat
# import numpy as np
# import os

# DATA_PATH = '/home/jpimen/projects/aip-frudzicz/jpimen/src/PLM_Graphs_Phase_2/data/'

# # https://pypi.org/project/textstat/
# def flesch_reading_ease_categorical(flesch_reading_score):
#   #flesch_reading_score = textstat.flesch_reading_ease(text)
#   if flesch_reading_score >= 90:
#     return 'Very Easy'
#   elif flesch_reading_score >= 80:
#     return 'Easy'
#   elif flesch_reading_score >= 70:
#     return 'Fairly Easy'
#   elif flesch_reading_score >= 60:
#     return 'Standard'
#   elif flesch_reading_score >= 50:
#     return 'Fairly Difficult'
#   elif flesch_reading_score >= 30:
#     return 'Difficult'
#   else:
#     return 'Very Confusing'

# def get_text_information(dataset, train, validation, test):
#   vectorizer = CountVectorizer() 
  
#   X_train = vectorizer.fit_transform(train['text'])
#   train['length'] = X_train.sum(axis = 1)

#   X_validation = vectorizer.fit_transform(validation['text']) # fit_transform to count all words
#   validation['length'] = X_validation.sum(axis = 1)

#   X_test = vectorizer.fit_transform(test['text'])
#   test['length'] = X_test.sum(axis = 1)

#   print('Text length', flush = True)
#   print('Train', train['length'].min(), train['length'].mean(), '±', train['length'].std(), train['length'].max(), flush = True)
#   print('Validation', validation['length'].min(), validation['length'].mean(), '±', validation['length'].std(), validation['length'].max(), flush = True)
#   print('Test', test['length'].min(), test['length'].mean(), '±', test['length'].std(), test['length'].max(), flush = True)
#   print('', flush = True)

#   train['unique_words'] = np.array((X_train > 0).sum(axis = 1)).flatten()
#   validation['unique_words'] = np.array((X_validation > 0).sum(axis = 1)).flatten()
#   test['unique_words'] = np.array((X_test > 0).sum(axis = 1)).flatten()

#   train['type_token_ratio'] = train['unique_words'] / train['length']
#   validation['type_token_ratio'] = validation['unique_words'] / validation['length']
#   test['type_token_ratio'] = test['unique_words'] / test['length']
  
#   print('Type-token ratio', flush = True)
#   print('Train', train['type_token_ratio'].min(), train['type_token_ratio'].mean(), '±', train['type_token_ratio'].std(), train['type_token_ratio'].max(), flush = True)
#   print('Validation', validation['type_token_ratio'].min(), validation['type_token_ratio'].mean(), '±', validation['type_token_ratio'].std(), validation['type_token_ratio'].max(), flush = True)
#   print('Test', test['type_token_ratio'].min(), test['type_token_ratio'].mean(), '±', test['type_token_ratio'].std(), test['type_token_ratio'].max(), flush = True)
#   print('', flush = True)

#   train['Flesch_Reading_ease_score'] = train['text'].apply(textstat.flesch_reading_ease)
#   validation['Flesch_Reading_ease_score'] = validation['text'].apply(textstat.flesch_reading_ease)
#   test['Flesch_Reading_ease_score'] = test['text'].apply(textstat.flesch_reading_ease)

#   print('Flesch-Reading ease score', flush = True)
#   print('Train', train['Flesch_Reading_ease_score'].min(), train['Flesch_Reading_ease_score'].mean(), '±', train['Flesch_Reading_ease_score'].std(), train['Flesch_Reading_ease_score'].max(), flesch_reading_ease_categorical(train['Flesch_Reading_ease_score'].median()), train['Flesch_Reading_ease_score'].median(), flush = True)
#   print('Validation', validation['Flesch_Reading_ease_score'].min(), validation['Flesch_Reading_ease_score'].mean(), '±', validation['Flesch_Reading_ease_score'].std(), validation['Flesch_Reading_ease_score'].max(), flesch_reading_ease_categorical(validation['Flesch_Reading_ease_score'].median()), validation['Flesch_Reading_ease_score'].median(), flush = True)
#   print('Test', test['Flesch_Reading_ease_score'].min(), test['Flesch_Reading_ease_score'].mean(), '±', test['Flesch_Reading_ease_score'].std(), test['Flesch_Reading_ease_score'].max(), flesch_reading_ease_categorical(test['Flesch_Reading_ease_score'].median()), test['Flesch_Reading_ease_score'].median(), flush = True)
#   print('', flush = True)

#   train['average_syllables_per_word'] = train['text'].apply(textstat.syllable_count) / train['length']
#   validation['average_syllables_per_word'] = validation['text'].apply(textstat.syllable_count) / validation['length']
#   test['average_syllables_per_word'] = test['text'].apply(textstat.syllable_count) / test['length']

#   print('Average syllables per word', flush = True)
#   print('Train', train['average_syllables_per_word'].min(), train['average_syllables_per_word'].mean(), '±', train['average_syllables_per_word'].std(), train['average_syllables_per_word'].max(), flush = True)
#   print('Validation', validation['average_syllables_per_word'].min(), validation['average_syllables_per_word'].mean(), '±', validation['average_syllables_per_word'].std(), validation['average_syllables_per_word'].max(), flush = True)
#   print('Test', test['average_syllables_per_word'].min(), test['average_syllables_per_word'].mean(), '±', test['average_syllables_per_word'].std(), test['average_syllables_per_word'].max(), flush = True)
#   print('', flush = True)

#   train['average_sentence_length'] = train['length'] / train['text'].apply(textstat.sentence_count)
#   validation['average_sentence_length'] = validation['length'] / validation['text'].apply(textstat.sentence_count)
#   test['average_sentence_length'] = test['length'] / test['text'].apply(textstat.sentence_count)

#   print('Average sentence length', flush = True)
#   print('Train', train['average_sentence_length'].min(), train['average_sentence_length'].mean(), '±', train['average_sentence_length'].std(), train['average_sentence_length'].max(), flush = True)
#   print('Validation', validation['average_sentence_length'].min(), validation['average_sentence_length'].mean(), '±', validation['average_sentence_length'].std(), validation['average_sentence_length'].max(), flush = True)
#   print('Test', test['average_sentence_length'].min(), test['average_sentence_length'].mean(), '±', test['average_sentence_length'].std(), test['average_sentence_length'].max(), flush = True)
#   print('', flush = True)

#   train['percentage_monosyllabic_words'] = train['text'].apply(textstat.monosyllabcount) / train['length']
#   validation['percentage_monosyllabic_words'] = validation['text'].apply(textstat.monosyllabcount) / validation['length']
#   test['percentage_monosyllabic_words'] = test['text'].apply(textstat.monosyllabcount) / test['length']

#   print('Percentage of monosyllabic words', flush = True)
#   print('Train', train['percentage_monosyllabic_words'].min(), train['percentage_monosyllabic_words'].mean(), '±', train['percentage_monosyllabic_words'].std(), train['percentage_monosyllabic_words'].max(), flush = True)
#   print('Validation', validation['percentage_monosyllabic_words'].min(), validation['percentage_monosyllabic_words'].mean(), '±', validation['percentage_monosyllabic_words'].std(), validation['percentage_monosyllabic_words'].max(), flush = True)
#   print('Test', test['percentage_monosyllabic_words'].min(), test['percentage_monosyllabic_words'].mean(), '±', test['percentage_monosyllabic_words'].std(), test['percentage_monosyllabic_words'].max(), flush = True)
#   print('', flush = True)

#   train['percentage_polysyllabic_words'] = train['text'].apply(textstat.polysyllabcount) / train['length']
#   validation['percentage_polysyllabic_words'] = validation['text'].apply(textstat.polysyllabcount) / validation['length']
#   test['percentage_polysyllabic_words'] = test['text'].apply(textstat.polysyllabcount) / test['length']

#   print('Percentage of polysyllabic words', flush = True)
#   print('Train', train['percentage_polysyllabic_words'].min(), train['percentage_polysyllabic_words'].mean(), '±', train['percentage_polysyllabic_words'].std(), train['percentage_polysyllabic_words'].max(), flush = True)
#   print('Validation', validation['percentage_polysyllabic_words'].min(), validation['percentage_polysyllabic_words'].mean(), '±', validation['percentage_polysyllabic_words'].std(), validation['percentage_polysyllabic_words'].max(), flush = True)
#   print('Test', test['percentage_polysyllabic_words'].min(), test['percentage_polysyllabic_words'].mean(), '±', test['percentage_polysyllabic_words'].std(), test['percentage_polysyllabic_words'].max(), flush = True)
#   print('', flush = True)

#   os.makedirs(os.path.join(DATA_PATH, 'text-properties', dataset), exist_ok = True)
#   train.drop(columns = ['text', 'label']).reset_index(names = 'index').to_csv(os.path.join(DATA_PATH, 'text-properties', dataset, 'train.csv'), index = False)
#   validation.drop(columns = ['text', 'label']).reset_index(names = 'index').to_csv(os.path.join(DATA_PATH, 'text-properties', dataset, 'validation.csv'), index = False)
#   test.drop(columns = ['text', 'label']).reset_index(names = 'index').to_csv(os.path.join(DATA_PATH, 'text-properties', dataset, 'test.csv'), index = False)

# if __name__ == '__main__':
#   for dataset, task in zip(['SST-2', 'SST-5', 'Ohsumed', 'CLEAR', 'IMDB-1k', 'Amazon-Video-Games'], ['classification', 'regression', 'classification', 'regression', 'classification', 'regression']):
#     print('-' * 30, f'{dataset} ({task})', '-' * 30, flush = True)
#     print('', flush = True)
    
#     get_text_information(
#       dataset,
#       pd.read_csv(os.path.join(DATA_PATH, task, dataset, 'train.csv')),
#       pd.read_csv(os.path.join(DATA_PATH, task, dataset, 'validation.csv')),
#       pd.read_csv(os.path.join(DATA_PATH, task, dataset, 'test.csv')),
#     )