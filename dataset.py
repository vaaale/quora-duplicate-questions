import csv
import os
import pickle
import re

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TRAIN_RAW = 'data/new_dataset.csv'
TEST_RAW = 'data/test.csv'
DATASET_CLEAN = 'data/data-clean.pkl'
TOKENIZER = 'data/tokenizer.pkl'
DATASET = 'data/dataset-{}.pkl'
WORCOUNT_DATASET = 'data/wordcount-dataset-{}.pkl'
TFIDF_DATASET = 'data/tfidf-dataset-{}.pkl'
EMBEDDING_DIM = 300


def generator(X, Y, batch_size):
    indices = list(range(len(X)))
    xa_batch = []
    xb_batch = []
    y_batch = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            xa_batch.append(X[i][0])
            xb_batch.append(X[i][1])
            y_batch.append(Y[i])
            if len(xa_batch) == batch_size:
                yield [np.asarray(xa_batch), np.asarray(xb_batch)], y_batch
                xa_batch = []
                xb_batch = []
                y_batch = []


def predict_generator(X, batch_size):
    indices = list(range(len(X)))
    xa_batch = []
    xb_batch = []
    for i in indices:
        xa_batch.append(X[i][0])
        xb_batch.append(X[i][1])
        if len(xa_batch) == batch_size or i == len(X) - 1:
            yield [np.asarray(xa_batch), np.asarray(xb_batch)]
            xa_batch = []
            xb_batch = []


def clean_text(text):
    # Clean the text
    text = re.sub(r"(www|http[s]?://)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "url", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\bm\b", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"(\d+)(th)", r"\g<1>", text)
    text = re.sub(r"(\d+)(nd)", r"\g<1>", text)
    text = re.sub(r"\be g\b", " eg ", text)
    text = re.sub(r"\bb g\b", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\b9 11\b", "911", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"\busa\b", " america ", text)
    text = re.sub(r"\bu s\b", " america ", text)
    text = re.sub(r"\bus\b", " america ", text)
    text = re.sub(r"\buk\b", " england ", text)
    text = re.sub(r"india", "india", text)
    text = re.sub(r"switzerland", "switzerland", text)
    text = re.sub(r"china", "china", text)
    text = re.sub(r"chinese", "chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"\bdms\b", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"\bcs\b", " computer science ", text)
    text = re.sub(r"\bupvotes\b", " up votes ", text)
    text = re.sub(r"\biphone\b", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"\bios\b", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"iii", "3", text)
    text = re.sub(r"the us", "america", text)
    text = re.sub(r"\bj k\b", " jk ", text)
    text = re.sub('[\d]+', 'num', text)
    return text


def load_training_data():
    train_q1 = []
    train_q2 = []
    labels = []
    print('Reading raw training data....')
    with open(TRAIN_RAW, encoding='iso8859-1') as f:
        next(f)
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        for segs in csv_reader:
            q1_clean = clean_text(segs[3].lower()).rstrip('\n')
            q2_clean = clean_text(segs[4].lower()).rstrip('\n')
            train_q1.append(q1_clean)
            train_q2.append(q2_clean)
            labels.append(segs[5])

    return train_q1, train_q2, labels


def load_validation_data():
    val_q1 = []
    val_q2 = []

    print('Reading raw test data....')
    with open(TEST_RAW, encoding='iso8859-1') as f:
        next(f)
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        for segs in csv_reader:
            q1_clean = clean_text(segs[1]).lower().rstrip('\n')
            q2_clean = clean_text(segs[2]).lower().rstrip('\n')
            val_q1.append(q1_clean)
            val_q2.append(q2_clean)

    return val_q1, val_q2


def fit_on_text(data):
    print('Analyzing dataset....')
    word_counts = {}
    lengths = []
    for sentence in data:
        words = sentence.split()
        if len(words) > 0:
            lengths.append(len(words))
            for w in words:
                if w in word_counts:
                    word_counts[w] += 1
                else:
                    word_counts[w] = 1

    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    return word_index, lengths


def build_embedding_matrix(embedding_dim, embedding_file, word_index):
    words = word_index.keys()

    print('Loading embeddings from {}'.format(embedding_file))
    embeddings_index = {}
    with open(embedding_file, encoding='iso8859-1') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in words:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        for w in word_index:
            if not w in embeddings_index:
                coefs = np.random.rand(embedding_dim) * 0.1
                embeddings_index[w] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('Embeddings shape: {}'.format(embedding_matrix.shape))

    return embedding_matrix, embeddings_index


def load_embedding_matrix(embedding_dim, embedding_file):
    if not os.path.isfile(TOKENIZER):
        q1, q2, labels, v1, v2 = load_data()
        all_text = q1 + q2 + v1 + v2
        word_index, statistiics = fit_on_text(all_text)
        embedding_matrix, embedding_index = build_embedding_matrix(embedding_dim, embedding_file, word_index)
        with open(TOKENIZER, 'wb') as f:
            tokenizer = {
                'word_index': word_index,
                'embedding_matrix': embedding_matrix,
                'embedding_index': embedding_index,
                'statistics': statistiics
            }
            pickle.dump(tokenizer, f)
    else:
        with open(TOKENIZER, 'rb') as f:
            tokenizer = pickle.load(f)
        word_index = tokenizer['word_index']
        embedding_matrix = tokenizer['embedding_matrix']
        embedding_index = tokenizer['embedding_index']
        statistiics = tokenizer['statistics']

    print('Max sentence length: {}'.format(max(statistiics)))
    print('Min sentence length: {}'.format(min(statistiics)))
    print('Mean sentence length: {}'.format(np.mean(np.array(statistiics))))
    print('Median sentence length: {}'.format(np.median(np.array(statistiics))))

    return embedding_matrix, embedding_index, word_index


def text_to_sequences(data, embeddings_index, max_len):
    print('Building sequences...')
    sequences = []
    for sentence in data:
        words = sentence.split()
        sent_len = len(words)
        pad_offset = max(0, max_len - sent_len)
        seq = np.zeros(max_len)
        for i in range(0, min(max_len, len(words))):
            w = words[i]
            seq[pad_offset + i] = embeddings_index[w]
        sequences.append(seq)
    return sequences


def load_data():
    if not os.path.isfile(DATASET_CLEAN):
        q1, q2, labels = load_training_data()
        v1, v2 = load_validation_data()
        all = (q1, q2, labels, v1, v2)
        # all = q1[0:2000]
        with open(DATASET_CLEAN, 'wb') as f:
            pickle.dump(all, f)
    else:
        with open(DATASET_CLEAN, 'rb') as f:
            all = pickle.load(f)
    return all


def dataset(seq_len=40):
    file_name = DATASET.format(seq_len)
    if not os.path.isfile(file_name):
        embedding_matrix, embedding_index, word_index = load_embedding_matrix(EMBEDDING_DIM, 'models/glove.840B.300d.txt')
        q1, q2, labels, v1, v2 = load_data()
        q1_seq = text_to_sequences(q1, word_index, seq_len)
        q2_seq = text_to_sequences(q2, word_index, seq_len)
        v1_seq = text_to_sequences(v1, word_index, seq_len)
        v2_seq = text_to_sequences(v2, word_index, seq_len)

        X_train = np.asarray(list(zip(q1_seq, q2_seq)))
        X_val = np.asarray(list(zip(v1_seq, v2_seq)))
        data = {
            'x_train': X_train,
            'x_val': X_val,
            'labels': labels
        }
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        X_train = data['x_train']
        labels = data['labels']
        X_val = data['x_val']
    return X_train, labels, X_val


def train_test_split(train_data, train_labels, test_split):
    nb_samples = len(train_data)
    idx = int(nb_samples * test_split)
    X_train = train_data[:-idx]
    y_train = train_labels[:-idx]
    X_test = train_data[-idx:]

    y_test = train_labels[-idx:]

    return X_train, y_train, X_test, y_test


def build_wordcount_dataset(vocab_size):
    filename = WORCOUNT_DATASET.format(vocab_size)
    if not os.path.isfile(filename):
        q1, q2, _, v1, v2 = load_data()
        texts = q1 + q2 + v1 + v2
        stop_words = stopwords.words('english') + ['why', 'where', 'who', 'what', 'when']
        vectorizer = CountVectorizer(stop_words=stop_words, max_features=vocab_size)
        X_sparse = vectorizer.fit_transform(texts)
        pickle.dump(X_sparse, open(filename, 'wb'))
    else:
        X_sparse = pickle.load(open(filename, 'rb'))

    return X_sparse


def build_tfidf_dataset(vocab_size):
    filename = TFIDF_DATASET.format(vocab_size)
    if not os.path.isfile(filename):
        q1, q2, _, v1, v2 = load_data()
        texts = q1 + q2 + v1 + v2
        stop_words = stopwords.words('english') + ['why', 'where', 'who', 'what', 'when']
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=vocab_size)
        X_sparse = vectorizer.fit_transform(texts)
        pickle.dump(X_sparse, open(filename, 'wb'))
    else:
        X_sparse = pickle.load(open(filename, 'rb'))

    return X_sparse

if __name__ == '__main__':

    # x_sparse = build_wordcount_dataset(10000)
    # print(x_sparse.shape)

    embedding_matrix, embedding_index, word_index = load_embedding_matrix(EMBEDDING_DIM, 'models/glove.840B.300d.txt')
    print('Embeddings shape: {}'.format(embedding_matrix.shape))

    X_train, labels, X_val = dataset(seq_len=40)
    print('Training data shape: {}'.format(X_train.shape))
    print('Validation data shape: {}'.format(X_val.shape))

