import glob
import os
import re

from keras.engine import Layer
from keras.layers import Input, LSTM, Embedding, Bidirectional, Conv1D, MaxPooling1D, K, initializers, \
    TimeDistributed, RepeatVector, Dense, Flatten, Dropout, Lambda, dot, Reshape, GaussianNoise, concatenate, \
    BatchNormalization
from keras.models import Model
from keras.models import model_from_json
from keras.regularizers import l2, l1
from keras.utils import vis_utils

from attentionwithcontext import AttentionWithContext
import tensorflow as tf
import numpy as np


class Attention(Layer):
    def __init__(self, **kwargs):
        """
        Attention operation for temporal data.
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.init((input_shape[-1],), name='{}_W'.format(self.name))
        self.b = K.ones((input_shape[1],), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W, self.b]

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W) + self.b)
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]


def build_model(model_dir='models', model_type=None, custom_objects=None, **kwargs):
    epoch = 0
    if model_type:
        print('Building model...')
        model = model_type(**kwargs)
    else:
        print('Loading model from json...')
        model = model_from_json(open('{}/model.json'.format(model_dir), 'r').read(), custom_objects=custom_objects)

    config_name = '{}/model.json'.format(model_dir)
    architecture_name = '{}/model.png'.format(model_dir)

    files = glob.glob(model_dir + '/*.hdf5')
    if len(files):
        files.sort(key=os.path.getmtime, reverse=True)
        print('Loading weights from: ' + files[0])
        if os.path.isfile(files[0]):
            epoch = int(re.findall(r'\d+', os.path.basename(files[0]))[0])
            model.load_weights(files[0], by_name=True)
    else:
        if type(model) == tuple:
            model[0].summary()
            vis_utils.plot_model(model[0], to_file=architecture_name, show_shapes=True)
            with open(config_name, 'w', encoding='iso8859-1') as model_out:
                model_out.write(model[0].to_json())
        else:
            model.summary()
            vis_utils.plot_model(model, to_file=architecture_name, show_shapes=True)
            with open(config_name, 'w', encoding='iso8859-1') as model_out:
                model_out.write(model.to_json())

    return model, epoch


def DeepAutoencoder(vocab_size):
    input = Input(shape=(vocab_size,))
    encoded = Dense(500, activation='sigmoid')(input)
    encoded = Dense(250, activation='sigmoid')(encoded)
    encoded = Dense(125, activation='sigmoid')(encoded)
    encoded = GaussianNoise(stddev=1.0)(encoded)
    decoded = Dense(2, activation='sigmoid')(encoded)
    decoded = Dense(125, activation='sigmoid')(decoded)
    decoded = Dense(250, activation='sigmoid')(decoded)
    decoded = Dense(500, activation='sigmoid')(decoded)
    output = Dense(vocab_size, activation='sigmoid')(decoded)

    model = Model(inputs=input, outputs=output)
    return model


def CNNLSTMBestSoFar(vocab_size, embedding_size, seq_len, embedding_matrix):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=False)
    conv1 = Conv1D(1000, 5, border_mode='valid', activation='relu')
    pool1 = MaxPooling1D(pool_length=4)
    shared_lstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM1', trainable=True))
    shared_lstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM2', trainable=True))
    shared_lstm_3 = Bidirectional(LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM3', trainable=True))

    embedding_a = embeddings(input_a)
    embedding_b = embeddings(input_b)
    a = conv1(embedding_a)
    b = conv1(embedding_b)
    a = pool1(a)
    b = pool1(b)
    a = shared_lstm_1(a)
    b = shared_lstm_1(b)
    a = shared_lstm_2(a)
    b = shared_lstm_2(b)
    a = shared_lstm_3(a)
    b = shared_lstm_3(b)
    # y = dot([a, b], mode='dot')
    y = Lambda(lambda x: similarity_score(x))([a, b])

    model = Model(input=[input_a, input_b], output=y)
    return model


def CNN(vocab_size, embedding_size, seq_len, embedding_matrix):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=False)

    conv1 = Conv1D(1024, 2, activation='tanh', padding='same')
    # pool1 = MaxPooling1D(2)
    conv2 = Conv1D(512, 2, activation='tanh', padding='same')
    # pool2 = MaxPooling1D(5)
    dense1 = Dense(512, activation='tanh', kernel_regularizer=l2())
    drop1 = Dropout(0.5)
    dense2 = Dense(128, activation='tanh', kernel_regularizer=l2())

    a = embeddings(input_a)
    a = conv1(a)
    # a = pool1(a)
    a = conv2(a)
    # a = pool2(a)
    a = Flatten()(a)
    a = dense1(a)
    a = drop1(a)
    a = dense2(a)

    b = embeddings(input_b)
    b = conv1(b)
    # b = pool1(b)
    b = conv2(b)
    # b = pool2(b)
    b = Flatten()(b)
    b = dense1(b)
    b = drop1(b)
    b = dense2(b)

    y = dot(inputs=[a, b], normalize=True, axes=1)
    y = Lambda(lambda x: 1 - (tf.acos(x) / np.pi))(y)

    model = Model(input=[input_a, input_b], output=y)
    return model


def AttentionLSTMBig(vocab_size, embedding_size, seq_len):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, input_length=seq_len, name='embeddings')
    shared_lstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout_U=0.3, dropout_W=0.3, U_regularizer=l2(), W_regularizer=l2(), activation='tanh'))
    shared_lstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout_U=0.3, dropout_W=0.3, U_regularizer=l2(), W_regularizer=l2(), activation='tanh'))
    shared_lstm_3 = Bidirectional(LSTM(64, return_sequences=True, dropout_U=0.3, dropout_W=0.3, U_regularizer=l2(), W_regularizer=l2(), activation='tanh'))
    att_a = AttentionWithContext()
    att_b = AttentionWithContext()

    embedding_a = embeddings(input_a)
    embedding_b = embeddings(input_b)
    a = shared_lstm_1(embedding_a)
    b = shared_lstm_1(embedding_b)
    a = shared_lstm_2(a)
    b = shared_lstm_2(b)
    a = shared_lstm_3(a)
    b = shared_lstm_3(b)
    a = att_a(a)
    b = att_b(b)
    y = dot([a, b], mode='dot')

    model = Model(input=[input_a, input_b], output=y)
    return model


def LSTMAutoEncoder1(vocab_size, embedding_size, seq_len, embedding_matrix):
    input = Input(shape=(seq_len,), dtype='int32', name='Sentence')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=True)(input)
    encoded = Conv1D(1000, 5, padding='same', activation='relu')(embeddings)
    encoded = Bidirectional(LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM1'))(encoded)
    encoded = Dense(128, activation='tanh')(encoded)
    decoded = RepeatVector(seq_len)(encoded)
    decoded = Bidirectional(LSTM(512, return_sequences=True, activation='tanh', name='DecoderLSTM1'))(decoded)
    decoded = Conv1D(1000, 5, padding='same', activation='relu')(decoded)
    # decoded = TimeDistributed(Conv1D(embedding_size, 1, padding='valid', activation='relu'))(decoded)
    decoded = TimeDistributed(Dense(embedding_size, activation='linear'))(decoded)
    model = Model(input=input, output=decoded)
    return model, embeddings


def LSTMHuge(vocab_size, embedding_size, seq_len, embedding_matrix):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=False)
    shared_lstm_1 = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM1', trainable=True)
    shared_lstm_2 = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM2', trainable=True, go_backwards=True)

    d = Dense(128, activation='tanh')

    embedding_a = embeddings(input_a)
    a = shared_lstm_1(embedding_a)
    a = shared_lstm_2(a)
    a = d(a)

    embedding_b = embeddings(input_b)
    b = shared_lstm_1(embedding_b)
    b = shared_lstm_2(b)
    b = d(b)

    y = dot([a, b], normalize=True, axes=1)
    y = Lambda(lambda x: 1 - (tf.acos(x) / np.pi))(y)

    model = Model(input=[input_a, input_b], output=y)
    return model


def similarity_score(x):
    a = x[0]
    b = x[1]
    return tf.exp(-(tf.reduce_sum(tf.abs(tf.subtract(a, b)), axis=1, keep_dims=True)))


def MaLSTM(vocab_size, embedding_size, seq_len, embedding_matrix):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=True)
    shared_lstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM1', trainable=True))
    shared_lstm_2 = Bidirectional(LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, recurrent_regularizer=l2(), activation='tanh', name='EncoderLSTM2', trainable=True))

    embedding_a = embeddings(input_a)
    a = shared_lstm_1(embedding_a)
    a = shared_lstm_2(a)

    embedding_b = embeddings(input_b)
    b = shared_lstm_1(embedding_b)
    b = shared_lstm_2(b)

    y = Lambda(lambda x: similarity_score(x))([a, b])

    model = Model(inputs=[input_a, input_b], outputs=y)
    return model


def AttentionLSTM(vocab_size, embedding_size, seq_len, embedding_matrix, dim_lstm, dim_dense):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')
    embeddings = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], input_length=seq_len, trainable=True)
    # shared_lstm_1 = Bidirectional(LSTM(dim_lstm, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, activity_regularizer=l2(), recurrent_regularizer=l2(), activation='tanh'))
    shared_lstm_1 = Bidirectional(LSTM(dim_lstm, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, activation='tanh'))
    shared_lstm_2 = Bidirectional(LSTM(dim_lstm, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, activation='tanh'))
    att = AttentionWithContext()

    embedding_a = embeddings(input_a)
    embedding_b = embeddings(input_b)
    a = shared_lstm_1(embedding_a)
    b = shared_lstm_1(embedding_b)
    a = shared_lstm_2(a)
    b = shared_lstm_2(b)
    y = concatenate([a, b], axis=1)
    y = att(y)
    y = Dense(dim_dense, activation='sigmoid')(y)
    y = Dense(1, activation='sigmoid')(y)

    model = Model(inputs=[input_a, input_b], outputs=y)
    return model

