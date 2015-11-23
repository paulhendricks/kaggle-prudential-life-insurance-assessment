from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of X (local validation)
    on the Kaggle Prudential Life Insurance challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4. Requires Scikit-Learn and Pandas.
    Recommended to run on GPU:
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch X: X
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
    Get the data from Kaggle: https://www.kaggle.com/c/prudential-life-insurance-assessment/data
'''


def load_data(path, train=True):
    df = pd.read_csv(path)
    if train:
        columns_of_interest = ['Id', 'Response', 'Ins_Age', 'Ht', 'Wt', 'BMI']
        df = df[columns_of_interest]
        labels = df['Response']
        df_1 = df.dropna(axis=0)
        del df_1['Id']
        del df_1['Response']
        X = df_1.values.copy().astype(np.float32)
        np.random.shuffle(X)  # https://youtu.be/uyUXoap67N8
        return X, labels
    else:
        columns_of_interest = ['Id', 'Ins_Age', 'Ht', 'Wt', 'BMI']
        df = df[columns_of_interest]
        ids = df['Id']
        del df['Id']
        X = df.values.copy().astype(np.float32)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

print("Loading data...")
X, labels = load_data('../data/prepped/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('../data/prepped/test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

n_hidden = 10

print("Building model...")

model = Sequential()
model.add(Dense(n_hidden, input_shape=(dims,)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_hidden))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_hidden))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam")

print("Training model...")
model.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.15)

print("Generating submission...")
proba = model.predict_proba(X_test)
proba = model.predict_proba(X)
response = [i.tolist().index(max(i)) for i in proba]
# make_submission(proba, ids, encoder, fname='../data/prepped/submission-neural-networks-1.csv')
# submission = pd.read_csv("../data/original/sample_submission.csv")
submission = pd.DataFrame({'Id': ids, 'Response': response}, columns = ['Id', 'Response'])
submission.to_csv('../data/prepped/submission-neural-networks-1.csv')