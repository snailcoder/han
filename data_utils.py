import re
import json
from collections import defaultdict
import numpy as np
from nltk import tokenize
from sklearn.model_selection import ShuffleSplit

NUM_YELP_LABEL = 5

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def stars2onehot(stars):
    assert stars in range(1, NUM_YELP_LABEL + 1)
    label = [0] * NUM_YELP_LABEL
    label[stars - 1] = 1
    return label

def load_data(review_path):
    all_label = []
    all_text = []
    vocab = defaultdict(int)
    with open(review_path, "r") as rfile:
        for l in rfile:
            rev = json.loads(l.strip())
            stars, text = rev.get("stars"), rev.get("text")
            if stars and text:
                label = stars2onehot(stars)
                all_label.append(label)
                text = clean_str(text)
                all_text.append(text)
                words = text.split()
                for w in words:
                    vocab[w] += 1
    return list(zip(all_text, all_label)), vocab

def replace_UNK(vocab, min_cnt):
    unk_vocab = defaultdict(int)
    for w in vocab:
        if vocab[w] >= min_cnt:
            unk_vocab[w] = vocab[w]
        else:
            unk_vocab["<UNK>"] += vocab[w]
    return unk_vocab

def get_word_idx_map(vocab):
    word_idx_map = {"<PAD>": 0}
    i = 1
    for w in vocab:
        word_idx_map[w] = i
        i += 1
    return word_idx_map

def doc2mat(doc, max_doc_len, max_sent_len, word_idx_map):
    mat = np.zeros([max_doc_len, max_sent_len], dtype=np.int32)
    sentences = tokenize.sent_tokenize(doc)
    i = 0
    while i < len(sentences) and i < max_doc_len:
        row = []
        words = sentences[i].split()
        j = 0
        while j < len(words) and j < max_sent_len:
            idx = word_idx_map.get(words[j], 0)
            row.append(idx)
        mat[i] = row
        i += 1
    return mat

def docs2mat(docs, max_doc_len, max_sent_len, word_idx_map):
    docs_mat = np.zeros([len(docs), max_doc_len, max_sent_len])
    i = 0
    while i < len(docs):
      dmat = doc2mat(docs[i], max_doc_len, max_sent_len, word_idx_map)
      docs_mat[i] = dmat
      i += 1
    return docs_mat

def split_train_test(docs, labels):
    folds = []
    rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train, test in rs.split(docs):
        train_data = np.array(zip(docs[train], labels[train]))
        test_data = np.array(zip(docs[test], labels[test]))
        folds.append((train_data, test_data))
    return folds