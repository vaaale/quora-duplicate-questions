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



def build_model(seq_len):
    input_a = Input(shape=(seq_len,), dtype='int32', name='Question_1')
    input_b = Input(shape=(seq_len,), dtype='int32', name='Question_2')

    shared_lstm_1 = Bidirectional(LSTM(300, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, activity_regularizer=l2(), recurrent_regularizer=l2(), activation='tanh'))
    att = AttentionWithContext()

    a = shared_lstm_1(input_a)
    b = shared_lstm_1(input_b)
    y = concatenate([a, b], axis=1)
    y = att(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)





