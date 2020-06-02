import gensim

from dataset import load_training_data, load_validation_data

print('Preparing data....')
train_q1, train_q2, labels = load_training_data()
val_q1, val_q2 = load_validation_data()
texts = train_q1 + train_q2 + val_q1 + val_q2
sequences = [sent.split() for sent in texts]
print('{} sentences.'.format(len(sequences)))

print('Training Word2Vec model...')
model = gensim.models.Word2Vec(size=100, window=5, min_count=2, iter=10)
model.build_vocab(sequences)  # can be a non-repeatable, 1-pass generator
model.train(sequences)  # can be a non-repeatable, 1-pass generator
model.wv.save_word2vec_format('models/embeddings-100.txt', 'models/vocab-200.txt')

model = gensim.models.Word2Vec(size=200, window=5, min_count=2, iter=10)
model.build_vocab(sequences)  # can be a non-repeatable, 1-pass generator
model.train(sequences)  # can be a non-repeatable, 1-pass generator
model.wv.save_word2vec_format('models/embeddings-200.txt', 'models/vocab-200.txt')
