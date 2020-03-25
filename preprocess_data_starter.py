import os
import sys
import operator
import time
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # needed by word_tokenize

# Assuming this file is put under the same parent directoray as the data directory, and the data directory is named "20news-train"
root_path = "./20news-train"
# The maximum size of the final vocabulary. It's a hyper-parameter. You can change it to see what value gives the best performance.
MAX_VOCAB_SIZE = 40000

start_time = time.time()
vocab_full = {}
n_doc = 0
# Only keep the data dictionaries and ignore possible system files like .DS_Store
folders = [os.path.join(root_path, name) for name in os.listdir(
    root_path) if os.path.isdir(os.path.join(root_path, name))]
for folder in folders:
    for filename in os.listdir(folder):
        file = os.path.join(folder, filename)
        n_doc += 1
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = word_tokenize(line)
                for token in tokens:
                    vocab_full[token] = vocab_full.get(token, 0) + 1
print(f'{n_doc} documents in total with a total vocab size of {len(vocab_full)}')
vocab_sorted = sorted(vocab_full.items(),
                      key=operator.itemgetter(1), reverse=True)
vocab_truncated = vocab_sorted[:MAX_VOCAB_SIZE]
# Save the vocabulary to file for visual inspection and possible analysis
with open('vocab.txt', 'w') as f:
    for vocab, freq in vocab_truncated:
        f.write(f'{vocab}\t{freq}\n')
# The final vocabulary is a dict mapping each token to its id. frequency information is not needed anymore.
vocab = dict([(token, id) for id, (token, _) in enumerate(vocab_truncated)])
# Since we have truncated the vocabulary, we will encounter many tokens that are not in the vocabulary. We will map all of them to the same 'UNK' token (a common practice in text processing), so we append it to the end of the vocabulary.
vocab['UNK'] = MAX_VOCAB_SIZE
vocab_size = len(vocab)
unk_id = MAX_VOCAB_SIZE
elapsed_time = time.time() - start_time
print(f'Vocabulary construction took {elapsed_time} seconds')

# Since we have truncated the vocabulary, it's now reasonable to hold the entire feature matrix in memory (it takes about 3.6GB on a 64-bit machine). If memory is an issue, you could make the vocabulary even smaller or use sparse matrix.
start_time = time.time()
features = np.zeros((n_doc, vocab_size), dtype=int)
print(f'The feature matrix takes {sys.getsizeof(features)} Bytes.')
# The class label of each document
labels = np.zeros(n_doc, dtype=int)
# The mapping from the name of each class label (i.e., the subdictionary name corresponding to a topic) to an integer ID
label2id = {}
label_id = 0
doc_id = 0
for folder in folders:
    label2id[folder] = label_id
    for filename in os.listdir(folder):
        labels[doc_id] = label_id
        file = os.path.join(folder, filename)
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                tokens = word_tokenize(line)
                for token in tokens:
                    # if the current token is in the vocabulary, get its ID; otherwise, get the ID of the UNK token
                    token_id = vocab.get(token, unk_id)
                    features[doc_id, token_id] += 1
        doc_id += 1
    label_id += 1
elapsed_time = time.time() - start_time
print(f'Feature extraction took {elapsed_time} seconds')
