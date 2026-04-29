import os
import torch
import transformers
import pandas as pd
import typing

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STORAGE_PATH = '../data/chunked_documents/'
os.makedirs(STORAGE_PATH, exist_ok = True)

def chunk_and_stride(
    text : str,
    label : int,
    identifier : int,
    #
    chunk_size : int,
    left_stride : int,
    right_stride : int,
    #
    tokenizer,
    maximum_chunk_size : int,
    #
    original_special_tokens : typing.List[int],
    #
    device,
  ):
  
  if maximum_chunk_size < chunk_size:
    raise ValueError('The selected chunk size is not accepted by the pre-trained model.')

  if left_stride + right_stride >= chunk_size:
    raise ValueError('The selected strides are greater or equal to the total chunk size.')
    
  encoding = tokenizer(
    text,
    return_tensors = 'pt',
    padding = False,
    truncation = False
  )
  input_identifiers = encoding['input_ids'].to(device)

  document_length = input_identifiers.size(1)

  c = chunk_size - left_stride - right_stride

  chunks = list()

  # Single chunk when document fits inside the context window
  if document_length <= maximum_chunk_size:
    left_stride = 0
    right_stride = 0
    c = document_length  

  # Compute the information (attention coefficients and embeddings) for each chunk
  s = 1
  chunk_identifier = 0
  while s < document_length - 1:
    l = max(1, s - left_stride)
    end_index = s + c - 2 if l > 1 else s + c + left_stride - 2
    r = min(document_length - 1, end_index + right_stride)
    
    # Chunk the input sequences and add special tokens to start and end of each chunk
    chunk_input_identifiers = torch.cat((input_identifiers[:, 0].reshape(1, 1), input_identifiers[:, l : r], input_identifiers[:, document_length - 1].reshape(1, 1)), dim = 1)    
    # chunk_text = tokenizer.decode(chunk_input_identifiers[0], skip_special_tokens = True).replace('##', '')
    chunk_text = tokenizer.decode(chunk_input_identifiers[0][~torch.isin(chunk_input_identifiers[0], torch.tensor(original_special_tokens, device = device))], skip_special_tokens = False).replace('##', '')
    
    chunks.append({
      'identifier' : identifier,
      'chunk' : chunk_identifier,
      'label' : label,
      'text' : chunk_text
    })

    s = end_index - 1

    chunk_identifier += 1

    if r == document_length - 1:
      break
  
  return chunks

def chunk_and_stride_all_documents(
    dataset,
    split,
    #
    chunk_size,
    left_stride,
    right_stride,
    #
    tokenizer,
    maximum_chunk_size,
    #
    original_special_tokens,
    #
    device
  ):
  
  df = pd.read_csv(f'../data/with_validation_splits/{dataset}/{split}.csv')

  documents = list()
  for row in df.itertuples():
    chunks = chunk_and_stride(
      text = row.text,
      label = row.label,
      identifier = row.Index,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      tokenizer = tokenizer,
      maximum_chunk_size = maximum_chunk_size,
      #
      original_special_tokens = original_special_tokens,
      #
      device = device,
    )
    documents.append(chunks)
  
  os.makedirs(os.path.join(STORAGE_PATH, dataset), exist_ok = True)

  pd.DataFrame([chunk for document_chunks in documents for chunk in document_chunks]) \
    .sample(frac = 1, random_state = SEED) \
    .reset_index(drop = True) \
    .to_csv(os.path.join(STORAGE_PATH, dataset, f'{split}.csv'), index = False)

if __name__ == '__main__':
  for dataset in ['IMDb', 'SST-2', 'Ohsumed', 'R8', 'IMDb-1k']:
    tokenizer = transformers.AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    original_special_tokens = tokenizer.all_special_ids
    for split in ['train', 'validation', 'test']:
      chunk_and_stride_all_documents(
        dataset = dataset,
        split = split,
        #
        chunk_size = 512,
        left_stride = 128,
        right_stride = 0,
        #
        tokenizer = tokenizer,
        maximum_chunk_size = 512,
        #
        original_special_tokens = original_special_tokens,
        #
        device = DEVICE,
      )