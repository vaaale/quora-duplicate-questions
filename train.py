import datetime
import os

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import RMSprop, Adam

from dataset import generator, dataset, train_test_split, load_embedding_matrix, EMBEDDING_DIM
from model import build_model, LSTMHuge, CNN, MaLSTM, CNNLSTMBestSoFar, AttentionLSTM
from tbembeddings import TensorBoard
import keras.backend as K

SEQ_LENGTH = 30
LOG_DIR = 'logs'
MODEL_DIR = 'models'
BATCH_SIZE = 256

embedding_matrix, embedding_index, word_index = load_embedding_matrix(EMBEDDING_DIM, 'models/glove.840B.300d.txt')
train, labels, val = dataset(seq_len=SEQ_LENGTH)
x_train, y_train, x_test, y_test = train_test_split(train, labels, test_split=0.2)

train_gen = generator(x_train, y_train, BATCH_SIZE)
test_gen = generator(x_test, y_test, BATCH_SIZE)
# val_gen = generator(val, np.zeros(len(val), 1), 10)

# nb_train_batches = 100
# nb_test_batches = 10
nb_train_batches = int(np.floor(len(x_train)/BATCH_SIZE))
nb_test_batches = int(np.floor(len(x_test)/BATCH_SIZE))

print('Total number of training samples: {}'.format(len(x_train)))
print('Number of training batches pr. epoch: {}'.format(nb_train_batches))
print('Number of test batches pr. epoch: {}'.format(nb_test_batches))


def init(run, prefix=None):
    run_dir = '{}/{}{}'.format(MODEL_DIR, prefix if prefix else '', run)
    log_dir = '{}/{}{}'.format(LOG_DIR, prefix if prefix else '', run)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return run_dir, log_dir

# run_dir, log_dir = init('AttLSTM-20170423-1638')
run_dir, log_dir = init(datetime.datetime.now().strftime('%Y%m%d-%H%M'), prefix='AttLSTM-')
K.set_learning_phase(True)
model, start_epoch = build_model(run_dir, AttentionLSTM,
                                 vocab_size=embedding_matrix.shape[0],
                                 embedding_size=embedding_matrix.shape[1],
                                 seq_len=SEQ_LENGTH,
                                 embedding_matrix=embedding_matrix,
                                 dim_lstm=100, dim_dense=50)
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])

tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=1)
checkpoint = ModelCheckpoint(run_dir + '/model-{epoch:02d}-{val_loss:.6f}-{val_acc:.6f}.hdf5', monitor='val_loss',
                             save_best_only=True,
                             mode='min', save_weights_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4, min_lr=0.00001)


class Sample(Callback):

    def __init__(self, word_index):
        super().__init__()
        self.word_index = word_index
        self.index_word = {val: key for (key, val) in word_index.items()}

    def build_sentences(self, x):
        result = []
        qb1 = x[0]
        qb2 = x[1]
        for q1, q2 in zip(qb1, qb2):
            words1 = ' '.join([self.index_word[w] for w in q1 if w])
            words2 = ' '.join([self.index_word[w] for w in q2 if w])
            result.append([words1, words2])
        return result

    def on_epoch_end(self, epoch, logs=None):
        batch = next(test_gen)
        x_b = batch[0]
        truth = batch[1]
        sentences = self.build_sentences(x_b)
        pred = model.predict_on_batch(x_b).squeeze()
        print('\nSample,  Prediction, Truth, Q1, Q2')
        for i in range(10):
            print('Sample: {}, {:.2f}, {},[{}], [{}]'.format(i, pred[i], truth[i], sentences[i][0], sentences[i][1]))
        print('.\n')


model.fit_generator(train_gen,
                    validation_data=test_gen,
                    steps_per_epoch=nb_train_batches,
                    validation_steps=nb_test_batches,
                    epochs=50,
                    initial_epoch=start_epoch + 1 if start_epoch > 0 else start_epoch,
                    callbacks=[checkpoint, tb, reduce_lr, Sample(word_index)],
                    verbose=1)
